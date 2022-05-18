# coding=utf-8
# Copyright 2018 The Microsoft Research Asia Locallm Team Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Locallm model. """


import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig
from transformers.models.layoutlm import modeling_layoutlm
from transformers.models.layoutlm.modeling_layoutlm import *


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.

    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension

        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence

        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class LayoutLMCRFForTokenClassification(modeling_layoutlm.LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(config.hidden_size, self.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(modeling_layoutlm.LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=modeling_layoutlm._CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> token_labels = torch.tensor([1,1,0,0]).unsqueeze(0) # batch size of 1

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=token_labels)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        mask_crf = input_ids > -1
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        _, logits = self.crf(sequence_output, mask_crf)
        logits = nn.functional.one_hot(torch.tensor(logits))
        
        loss = None
        if labels is not None:
            mask_crf = labels != -100
            labels[labels==-100] = 0
            loss = self.crf.loss(sequence_output, labels, mask_crf)
            # loss_fct = CrossEntropyLoss()

            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
