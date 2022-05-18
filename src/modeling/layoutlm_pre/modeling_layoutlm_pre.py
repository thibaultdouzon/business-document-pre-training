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


class LayoutLMPreTrainPredictionHead(nn.Module):
    def __init__(self, config, out_size):
        super().__init__()
        self.transform = LayoutLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, out_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->LayoutLM
class LayoutLMPreTrainHead(nn.Module):
    def __init__(self, config, out_size):
        super().__init__()
        self.predictions = LayoutLMPreTrainPredictionHead(config, out_size)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@add_start_docstrings("""Layoutlm Model with a `language modeling` head on top. """, modeling_layoutlm.LAYOUTLM_START_DOCSTRING)
class LayoutLMPreTrainForMultiTask(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.layoutlm = LayoutLMModel(config)
        self.cls = nn.ModuleList([
            LayoutLMPreTrainHead(config, out_size)
            for out_size in (config.vocab_size, 3, 2)
            ])
        
        self.task_out_size = [self.config.vocab_size, 3, 2]

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def get_output_embeddings(self, i=0):
        return self.cls[i].predictions.decoder

    def set_output_embeddings(self, new_embeddings, i=0):
        self.cls[i].predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(modeling_layoutlm.LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=modeling_layoutlm._CONFIG_FOR_DOC)
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
        task_id=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import LocallmTokenizer, LocallmForMaskedLM
            >>> import torch

            >>> tokenizer = LocallmTokenizer.from_pretrained('microsoft/locallm-base-uncased')
            >>> model = LocallmForMaskedLM.from_pretrained('microsoft/locallm-base-uncased')

            >>> words = ["Hello", "[MASK]"]
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

            >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=labels)

            >>> loss = outputs.loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        task_idx = task_id[0].item()
        prediction_scores = self.cls[task_idx](sequence_output)

        if not torch.all(task_id == task_idx):
            if not return_dict:
                return (torch.zeros(1, device=sequence_output.device), prediction_scores, outputs[2:])
            else:
                return MaskedLMOutput(
                    loss=torch.zeros(1, device=sequence_output.device),
                    logits=prediction_scores,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.task_out_size[task_idx]),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )