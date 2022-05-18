from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, ClassVar, Union

import numpy as np
import torch

from torch.utils.data import Dataset
from torchtyping import TensorType
from transformers import LayoutLMTokenizerFast

from src import documents, label_config


DEFAULT_TOKENIZER: LayoutLMTokenizerFast = LayoutLMTokenizerFast.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)

@dataclass
class BIESOTag:
    """Defines BIESO Tags over a set of labels
    Meanings of each tag:
    * B: begin of a new occurence
    * I: inside of current occurence
    * E: end of occurence
    * S: single token occurence
    * O: outside of any occurence
    Tags B,I,E,S are paired with a label to form a BIESOTag, O is always single (paired with OTHER only)
    """
    tag_l: ClassVar[list[Union[Literal["B"], Literal["I"], Literal["E"], Literal["S"]]]] = ["B", "I", "E", "S"]
    tag_mapping_to_int: ClassVar[dict[str, int]] = dict(zip(tag_l, range(4)))

    label: label_config.TaskLabel
    tag: Union[Literal["B"], Literal["I"], Literal["E"], Literal["S"], Literal["O"]]

    def __str__(self):
        if self.label == label_config.TaskLabel.OTHER:
            return "O"
        else:
            return f"{self.tag}-{self.label.name.upper()}"

    def __int__(self):
        # Other is always 0
        if self.label == label_config.TaskLabel.OTHER:
            return 0
        else:
            assert self.tag != "O"
            return 1 + (self.label.value - 1) * len(self.tag_l) + self.tag_mapping_to_int[self.tag]

    @classmethod
    def from_int(cls, value: int) -> "BIESOTag":
        """Compute BIESOTag instance from a given int value.

        Args:
            value (int): given int value corresponding to a tag

        Raises:
            ValueError: if value does not match to any tag

        Returns:
            BIESOTag: corresponding BIESOTag if exists
        """
        if value < 0 or value >= NUM_BIESO_TAGS:
            raise ValueError(f"Value must be between 0 and {NUM_BIESO_TAGS} included. Found {value}.")

        if value == 0:
            return cls(label_config.TaskLabel.OTHER, "O")
        else:
            lbl_id, tag_id = divmod(value - 1, len(cls.tag_l))
            return cls(label_config.TaskLabel(lbl_id + 1), cls.tag_l[tag_id])


NUM_BIESO_TAGS = (len(label_config.TaskLabel) - 1) * len(BIESOTag.tag_l) + 1 

class SpanTag(int, Enum):
    OTHER  = 0
    BEGIN  = 1
    INNER  = 2
    END    = 3
    SINGLE = 4

    def __str__(self):
        return self.name.upper()

    def __int__(self):
        return self.value

NUM_OUT_TAGS = NUM_BIESO_TAGS if label_config.SelectedTask == label_config.TaskType.SEQUENCE_TAGGING else \
               len(SpanTag)   if label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING else \
               NotImplementedError()


def get_bieso_tags_from_labels(labels: list[int]) -> list[int]:
    """Convert list of labels to a list of BIESOTag corresponding to the input labeling
    Input and output lists are the same size.

    Args:
        labels (list[int]): list of ints corresponding to Labels

    Returns:
        list[int]: list of ints corresponding to BIESOTags
    """
    bieso_tags = [0]
    for lbl_value, lbl_value_next in zip(labels, labels[1:] + [0]):
        if lbl_value == 0:
            bieso_tags.append(int(BIESOTag(label_config.TaskLabel.OTHER, "O")))

        elif BIESOTag.from_int(bieso_tags[-1]).label.value != lbl_value:
            # B ou S
            if lbl_value == lbl_value_next:
                bieso_tags.append(int(BIESOTag(label_config.TaskLabel(lbl_value), "B")))
            else:
                bieso_tags.append(int(BIESOTag(label_config.TaskLabel(lbl_value), "S")))
        else:
            # I ou E
            if lbl_value == lbl_value_next:
                bieso_tags.append(int(BIESOTag(label_config.TaskLabel(lbl_value), "I")))
            else:
                bieso_tags.append(int(BIESOTag(label_config.TaskLabel(lbl_value), "E")))
    
    return bieso_tags[1:]


def get_span_from_bieso_tag(tags: list[int], label: label_config.TaskLabel) -> list[int]:
    span_tag = []
    for tag_value in tags:
        tag = BIESOTag.from_int(tag_value)
        if tag.label == label:
            if tag.tag == "B":  # begin
                span_tag.append(SpanTag.BEGIN.value)
            elif tag.tag == "E":  # end
                span_tag.append(SpanTag.END.value)
            elif tag.tag == "S":  # single
                span_tag.append(SpanTag.SINGLE.value)
            else:  # must be inner
                span_tag.append(SpanTag.INNER.value)
        else:  # any other tag / label
            span_tag.append(SpanTag.OTHER.value)
    return span_tag


@dataclass
class Encoding:
    """Proxy class to add types to the various tensors composing an input
    Implements python dict protocol (keys, values, items)
    """

    input_ids: TensorType["sequence", int]
    token_type_ids: TensorType["sequence", int]
    attention_mask: TensorType["sequence", int]
    bbox: TensorType["sequence", 4, int]
    offset_mapping: TensorType["sequence", 2, int]
    labels: TensorType["sequence", int]
    filename: str
    part: int

    def __getitem__(self, key):
        return getattr(self, key)

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
            ]
        }

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

@dataclass
class SpanQAEncoding(Encoding):
    qa_label: label_config.TaskLabel
    


def encode_one_qa(doc: documents.LabeledDocument, label):
    pass
    # TODO for predictive results


class DocumentDataset(Dataset):
    """DocumentDataset holds a dataset of OCR document with labels and provides huggingface friendly data"""

    def __init__(
        self,
        document_l: list[documents.LabeledDocument],
        tokenizer: LayoutLMTokenizerFast = DEFAULT_TOKENIZER,
        task_type: label_config.TaskType = label_config.SelectedTask,
        ignore_non_beg_token: bool = True
    ):
        # ic(document_l)
        self.documents = document_l
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.ignore_non_beg_token = ignore_non_beg_token

        if task_type == label_config.TaskType.SEQUENCE_TAGGING:
            self._input_encodings, self._label_encodings, self._filenames, self._parts = self._sequence_tagging_get_encodings()
        elif task_type == label_config.TaskType.SPAN_QUESTION_ANSWERING:
            self._input_encodings, self._label_encodings, self._filenames, self._parts, self._qa_labels = self._span_question_anwsering_get_encodings()
            
    @property
    def input_encodings(self):
        return self._input_encodings

    @property
    def label_encodings(self):
        return self._label_encodings

    #pylint: disable=not-callable
    def __getitem__(self, idx) -> Encoding:
        item = {
            key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()
        }
        item["labels"] = torch.tensor(self.label_encodings[idx])
        item["filename"] = self._filenames[idx]
        item["part"] = self._parts[idx]
        if self.task_type == label_config.TaskType.SPAN_QUESTION_ANSWERING:
            return SpanQAEncoding(**item, qa_label=self._qa_labels[idx])
        else:
            return Encoding(**item)
    #pylint: enable=not-callable

    def __len__(self) -> int:
        return len(self.label_encodings)

    def _sequence_tagging_get_encodings(self) -> tuple[dict[str, list[list]], list[list]]:
        """Compute encodings for each document in the dataset.
        Resulting encoding size does not exceed 512 tokens

        Returns:
            tuple[dict[str, list[list]], list[list]]: input encodings as a dictionary and corresponding labels per token
        """
        # ic(self.documents)
        texts = [doc.document.words for doc in self.documents]
        input_encodings = self.tokenizer(
            texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )

        labels = [[label.value for label in doc.labels] for doc in self.documents]
        filenames = [doc.document.filename for doc in self.documents]
        parts = [doc.part for doc in self.documents]
        

        label_tags = [get_bieso_tags_from_labels(doc_labels) for doc_labels in labels]

        positions = [list(map(list, doc.document.positions)) for doc in self.documents]

        position_encodings = []
        label_encodings: list[list[int]] = []
        for per_doc_labels, per_doc_positions, doc_offset in zip(
            label_tags, positions, input_encodings.offset_mapping
        ):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            doc_enc_pos = np.ones((len(doc_offset), 4), dtype=int) * 1000
            arr_offset = np.array(doc_offset)

            start_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            (sequel_indices,) = (
                arr_offset[:, 0] != 0
            ).nonzero()  # This is a np array, nonzero returns a tuple, one 1D array per dim


            doc_enc_labels[start_mask] = per_doc_labels[: start_mask.sum()]

            # for coord in sequel_indices:
            #     if doc_enc_labels[coord - 1] > 0:
            #         doc_enc_labels[coord] = doc_enc_labels[coord - 1]

            
            doc_enc_pos[start_mask] = per_doc_positions[: start_mask.sum()]
            
            for coord in sequel_indices:
                if self.ignore_non_beg_token:
                    doc_enc_labels[coord] = doc_enc_labels[coord - 1]
                doc_enc_pos[coord] = doc_enc_pos[coord - 1]

            label_encodings.append(doc_enc_labels.tolist())
            position_encodings.append(doc_enc_pos.tolist())

        input_encodings["bbox"] = position_encodings
        # input_encodings["filename"] = [doc.document.filename for doc in self.documents]

        return input_encodings, label_encodings, filenames, parts

    def _span_question_anwsering_get_encodings(self) -> tuple[dict[str, list[list]], list[list]]:
        # TODO: not memory efficient better way ?
        qa_labels = [lbl for _ in self.documents for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]
        text_labels = [[lbl.name] for _ in self.documents for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]
        texts = [doc.document.words for doc in self.documents for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]
        filenames = [doc.document.filename for doc in self.documents for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]
        parts = [doc.part for doc in self.documents for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]

        input_encodings = self.tokenizer(
            text=text_labels,
            text_pair=texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        
        labels = [[label.value for label in doc.labels] for doc in self.documents]
        

        label_spans = [get_span_from_bieso_tag(get_bieso_tags_from_labels(doc_labels), lbl) for doc_labels in labels for lbl in label_config.TaskLabel if lbl != label_config.TaskLabel.OTHER]

        positions = [list(map(list, doc.document.positions)) for doc in self.documents]

        position_encodings = []
        label_encodings: list[list[int]] = []
        n_labels = sum(1 for _ in label_config.TaskLabel) - 1  # Remove OTHER label

        
        for doc_idx, (doc_offset, per_doc_label_span) in enumerate(zip(input_encodings.offset_mapping, label_spans)):
            per_doc_positions = positions[doc_idx // n_labels]
            context_beg = next(k for k, offset in enumerate(doc_offset[1:], 1) if offset == (0, 0)) + 1
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            doc_enc_pos = np.ones((len(doc_offset), 4), dtype=int) * 1000
            arr_offset = np.array(doc_offset)

            start_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            start_mask[:context_beg] = 0
            (sequel_indices,) = (
                arr_offset[:, 0] != 0
            ).nonzero()  # This is a np array, nonzero returns a tuple, one 1D array per dim

            doc_enc_labels[start_mask] = per_doc_label_span[: start_mask.sum()]
            if all(doc_enc_labels[start_mask] == 0):
                doc_enc_labels[0] = 4
            else:
                doc_enc_labels[0] = 0
            # for coord in sequel_indices:
            #     if doc_enc_labels[coord - 1] > 0:
            #         doc_enc_labels[coord] = doc_enc_labels[coord - 1]
            label_encodings.append(doc_enc_labels.tolist())

            doc_enc_pos[start_mask] = per_doc_positions[: start_mask.sum()]
            for coord in sequel_indices:
                doc_enc_pos[coord] = doc_enc_pos[coord - 1]

            position_encodings.append(doc_enc_pos.tolist())
        input_encodings["bbox"] = position_encodings

        return input_encodings, label_encodings, filenames, parts, qa_labels


def get_layoutlm_style_dataset(directory: Path) -> DocumentDataset:
    return DocumentDataset(sorted(
        documents.load_from_layoutlm_style_dataset(directory, "train") + documents.load_from_layoutlm_style_dataset(directory, "dev"),
        key=lambda doc: str(doc.document.filename)
    ))
