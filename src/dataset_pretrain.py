import itertools
import random

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import LayoutLMTokenizerFast

from src import dataset_util, documents, label_config, parser, icdar_sroie


def clamp(x, mi, ma):
    if x < mi:
        return mi
    if x > ma:
        return ma
    return x

def create_token_label_mask(words_idx, words_lbl, offset_mapping, question_offset=0):
    token_lbl_mask = np.full(len(offset_mapping), fill_value=-100, dtype=np.int64)
    idx2lbl = dict(zip(words_idx, words_lbl))
    word_counter = -(1 + question_offset)
    for i, offset in enumerate(offset_mapping):
        if offset[0] == 0 and offset[1] > 0:
            word_counter += 1
            if word_counter in idx2lbl:
                token_lbl_mask[i] = idx2lbl[word_counter]
    return token_lbl_mask


def read_pretrain_document_file(doc_name: Path, limit: int = 512):
    words, boxes = [], []
    with doc_name.open("r", encoding="utf-8") as fp:
        for line in fp.readlines():
            try:
                left, top, right, bottom, *word = line.strip().split()
                left, top, right, bottom = map(int, (left, top, right, bottom))
                left, top, right, bottom = map(lambda x: clamp(x, 0, 1000), (left, top, right, bottom))
                right = max(left, right)
                bottom = max(top, bottom)
                word = " ".join(w.strip() for w in word)

                words.append(word)
                boxes.append([left, top, right, bottom])
            except:
                pass
    beg = 0
    if len(words) > limit:
        n = len(words)
        beg = random.randint(0, n-limit)
    return words[beg:beg+limit], boxes[beg:beg+limit]


class PreTrainTaskType(int, Enum):
    MASK_LANG_MODEL  = 0
    NUM_ORDER  = 1
    LAYOUT  = 2

    def __str__(self):
        return self.name.upper()

    def __int__(self):
        return self.value


@dataclass
class MultiTaskEncoding(dataset_util.Encoding):
    task_id: PreTrainTaskType

    @property
    def __dict__(self):
        return {
            name: self[name]
            for name in [
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "bbox",
                "labels",
                "task_id"
            ]
        }

class PreTrainDocumentDatasetMLM(Dataset):
    """DocumentDataset holds a dataset of OCR document with labels and provides huggingface friendly data"""

    def __init__(
        self,
        document_l: list[Path],
        tokenizer: LayoutLMTokenizerFast = dataset_util.DEFAULT_TOKENIZER,
        provide_task_id: bool = False,
    ):
        self.document_l = document_l
        self.tokenizer = tokenizer
        self.provide_task_id = provide_task_id

    #pylint: disable=not-callable
    def __getitem__(self, idx) -> dataset_util.Encoding:
        item = {
            key: torch.tensor(val) for key, val in self._sequence_mlm_get_encodings(idx).items()
        }
        item["filename"] = self.document_l[idx].name
        item["part"] = 0
        if self.provide_task_id:
            item["task_id"] = int(PreTrainTaskType.MASK_LANG_MODEL)
        return MultiTaskEncoding(**item)
    #pylint: enable=not-callable

    def __len__(self) -> int:
        return len(self.document_l)
    
    def _sequence_mlm_get_encodings(self, idx: int):
        doc_name = self.document_l[idx]
        words, positions = read_pretrain_document_file(doc_name, limit=512)
        
        input_encodings = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        # Might break if chinese char appears
        
        filename = doc_name.name
        doc_offset = input_encodings.offset_mapping
        doc_inputs, modified_indices = self._random_replace_mlm(input_encodings.input_ids[:])

        doc_enc_labels = np.ones(len(doc_offset), dtype=np.int64) * -100
        doc_enc_labels[modified_indices] = np.array(input_encodings.input_ids)[modified_indices]
        doc_enc_pos = np.ones((len(doc_offset), 4), dtype=np.int64) * 1000
        
        doc_labels = doc_enc_labels
        
        arr_offset = np.array(doc_offset)
        start_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        (sequel_indices,) = (
            arr_offset[:, 0] != 0
        ).nonzero()  # This is a np array, nonzero returns a tuple, one 1D array per dim

        # DIRTY PLS FIX THIS
        if start_mask.sum() > len(positions):
            # Error chinese char.
            # Clean dataset before removing this
            return self._sequence_mlm_get_encodings((idx+1) % len(self))
        
        
        doc_enc_pos[start_mask] = positions[: start_mask.sum()]
        for coord in sequel_indices:
            doc_enc_pos[coord] = doc_enc_pos[coord - 1]

        position_encodings = doc_enc_pos

        input_encodings["bbox"] = position_encodings
        input_encodings["labels"] = doc_labels
        input_encodings["input_ids"] = doc_inputs

        return input_encodings
        
    def _random_replace_mlm(self, word_ids: list[int]) -> tuple[list[int], list[int]] :
        """In place modifications of token ids list for MLM

        Args:
            word_ids (list[int]): [description]
        """
        if self.tokenizer.pad_token_id in word_ids:
            n = word_ids.index(self.tokenizer.pad_token_id)
        else:
            n = len(word_ids)

        selected_ids = random.sample(range(n), int(n * 0.2))
        for some_id in selected_ids:
            rnd = random.random()
            if rnd < 0.8:
                # Replace token with __MASK__
                word_ids[some_id] = self.tokenizer.mask_token_id
            elif rnd < 0.9:
                # Replace token with a random one
                word_ids[some_id] = random.randrange(self.tokenizer.vocab_size)
            else:
                # Do not change the current token
                pass
        return word_ids, selected_ids

class PreTrainDocumentDatasetNumber(Dataset):
    def __init__(self, document_l: list[Path],
        tokenizer: LayoutLMTokenizerFast = dataset_util.DEFAULT_TOKENIZER,
    ):
        self.document_l = document_l
        self.tokenizer = tokenizer

    #pylint: disable=not-callable
    def __getitem__(self, idx) -> dataset_util.Encoding:
        item = {
            key: torch.tensor(val) for key, val in self._qa_number_get_encodings(idx).items()
        }
        item["filename"] = self.document_l[idx].name
        item["part"] = 0
        item["task_id"] = int(PreTrainTaskType.NUM_ORDER)
        return MultiTaskEncoding(**item)
    #pylint: enable=not-callable

    def __len__(self) -> int:
        return len(self.document_l)

    def _qa_number_get_encodings(self, idx: int):
        """Pretrain task where model needs to figure out which figures are bigger, smaller or equal to the one in the question

        Args:
            idx (int): [description]
        """
        doc_name = self.document_l[idx]
        words, positions = read_pretrain_document_file(doc_name, limit=512)
        word_starting_position = [0] + list(
            itertools.accumulate(map(lambda s: len(s) + 1, words))
        )[:-1]

        amounts = list(parser.amount_re.finditer(" ".join(words)))

        if len(amounts) == 0:
            return self._qa_number_get_encodings((idx+1) % len(self))

        question = self._gen_question(amounts)
        
        amounts_value = [parser.amount_parser(amount.group(0)) for amount in amounts]
        amounts_word_idx = [i for m in amounts for i in range(*icdar_sroie.retrieve_word_index_from_char_span(word_starting_position, m.span()))]
        question_value = parser.amount_parser(question)
        amounts_label = [0 if v < question_value else
                         1 if v == question_value else
                         2 for v in amounts_value]

        input_encodings = self.tokenizer(
            text=question.split(),
            text_pair=words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        # Might break if chinese char appears
        
        filename = doc_name.name
        doc_offset = input_encodings.offset_mapping

        doc_labels = create_token_label_mask(amounts_word_idx, amounts_label, doc_offset, question_offset=len(question.split()))
        # modify inputs here
        self._random_modify_inputs(input_encodings.input_ids, positions, doc_labels, doc_offset, question_offset=len(question.split()))
        
        arr_offset = np.array(doc_offset)
        start_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (np.array(input_encodings.token_type_ids) == 1)
        (sequel_indices,) = (
            arr_offset[:, 0] != 0
        ).nonzero()  # This is a np array, nonzero returns a tuple, one 1D array per dim

        # DIRTY PLS FIX THIS
        if start_mask.sum() > len(positions):
            # Error chinese char.
            # Clean dataset before removing this
            return self._qa_number_get_encodings((idx+1) % len(self))
        
        
        doc_enc_pos = np.ones((len(doc_offset), 4), dtype=np.int64) * 1000
        doc_enc_pos[start_mask] = positions[: start_mask.sum()]
        for coord in sequel_indices:
            doc_enc_pos[coord] = doc_enc_pos[coord - 1]

        position_encodings = doc_enc_pos

        input_encodings["bbox"] = position_encodings
        input_encodings["labels"] = doc_labels

        return input_encodings
    
    def _gen_question(self, matches):
        return random.choice(matches).group(0)
    
    def _random_modify_inputs(self, word_ids, positions, doc_labels, offset_mapping, question_offset=0):
        """In place modifications of token ids list for MLM

        Args:
            word_ids (list[int]): [description]
        """
        if self.tokenizer.pad_token_id in word_ids:
            n = word_ids.index(self.tokenizer.pad_token_id)
        else:
            n = len(word_ids)

        word_i = -(1 + question_offset)
        for token_i in range(n):
            if offset_mapping[token_i][0] == 0 and offset_mapping[token_i][1] > 0:
                word_i += 1
            if doc_labels[token_i] != -100:
                if random.random() < 0.15:  # 15% chance positions are wiped
                    positions[word_i] = [1000, 1000, 1000, 1000]
                if random.random() < 0.15:  # 15% chance positions are wiped
                    word_ids[token_i] = self.tokenizer.mask_token_id


class PreTrainDocumentDatasetLayout(Dataset):
    def __init__(self, document_l: list[Path],
        tokenizer: LayoutLMTokenizerFast = dataset_util.DEFAULT_TOKENIZER,
    ):
        self.document_l = document_l
        self.tokenizer = tokenizer

    #pylint: disable=not-callable
    def __getitem__(self, idx) -> dataset_util.Encoding:
        item = {
            key: torch.tensor(val) for key, val in self._qa_layout_get_encodings(idx).items()
        }
        item["filename"] = self.document_l[idx].name
        item["part"] = 0
        item["task_id"] = int(PreTrainTaskType.LAYOUT)
        return MultiTaskEncoding(**item)
    #pylint: enable=not-callable

    def __len__(self) -> int:
        return len(self.document_l)

    def _qa_layout_get_encodings(self, idx: int):
        """Pretrain task where model needs to figure out which tokens are inside a zone

        Args:
            idx (int): [description]
        """
        doc_name = self.document_l[idx]
        words, positions = read_pretrain_document_file(doc_name, limit=512)
        word_starting_position = [0] + list(
            itertools.accumulate(map(lambda s: len(s) + 1, words))
        )[:-1]

        pos_question = self._gen_question()
        
        words_label = [self._in_inside(pos, pos_question) for pos in positions]

        input_encodings = self.tokenizer(
            text=["position"],
            text_pair=words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        # Might break if chinese char appears
        
        doc_offset = input_encodings.offset_mapping

        doc_labels = create_token_label_mask(list(range(len(words))), words_label, doc_offset, question_offset=len(pos_question))
        # modify inputs here
        self._random_modify_inputs(input_encodings.input_ids, positions, doc_labels, doc_offset, question_offset=len(pos_question))
        
        arr_offset = np.array(doc_offset)
        start_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (np.array(input_encodings.token_type_ids) == 1)
        (sequel_indices,) = (
            arr_offset[:, 0] != 0
        ).nonzero()  # This is a np array, nonzero returns a tuple, one 1D array per dim

        # DIRTY PLS FIX THIS
        if start_mask.sum() > len(positions):
            # Error chinese char.
            # Clean dataset before removing this
            return self._qa_number_get_encodings((idx+1) % len(self))
        
        
        doc_enc_pos = np.ones((len(doc_offset), 4), dtype=np.int64) * 1000
        doc_enc_pos[1] = list(pos_question)
        doc_enc_pos[start_mask] = positions[: start_mask.sum()]
        for coord in sequel_indices:
            doc_enc_pos[coord] = doc_enc_pos[coord - 1]

        position_encodings = doc_enc_pos

        input_encodings["bbox"] = position_encodings
        input_encodings["labels"] = doc_labels

        return input_encodings
    
    def _gen_question(self):
        left = random.randint(0, 900)
        right = random.randint(left, 1000)

        top = random.randint(0, 900)
        bottom = random.randint(top, 1000)

        return  left, top, right, bottom
    
    def _in_inside(self, pos, pos_question):
        l, t, r, b = pos
        ll, tt, rr, bb = pos_question
        return 1 if ll < (l+r)/2 < rr and tt < (t+b)/2 < bb else 0

    def _random_modify_inputs(self, word_ids, positions, doc_labels, offset_mapping, question_offset=0):
        """In place modifications of token ids list for MLM

        Args:
            word_ids (list[int]): [description]
        """
        if self.tokenizer.pad_token_id in word_ids:
            n = word_ids.index(self.tokenizer.pad_token_id)
        else:
            n = len(word_ids)

        word_i = -(1 + question_offset)
        for token_i in range(n):
            if offset_mapping[token_i][0] == 0 and offset_mapping[token_i][1] > 0:
                word_i += 1
            if doc_labels[token_i] != -100:
                if random.random() < 0.15:  # 15% chance positions are wiped
                    positions[word_i] = [1000, 1000, 1000, 1000]
                if random.random() < 0.15:  # 15% chance positions are wiped
                    word_ids[token_i] = self.tokenizer.mask_token_id