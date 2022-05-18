import argparse
import enum
import json
import math

from collections import defaultdict
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any

import torch
import transformers

from torchtyping import TensorType
from tqdm import tqdm

from src import dataset_util, label_config, documents


CONFIDENCE_THRESHOLD_DP = 3.
CONFIDENCE_THRESHOLD_PRED = 5.


@dataclass
class MetricResult:
    recall: float
    precision: float
    f1: float = field(init=False)

    def __post_init__(self):
        self.f1 = (
            (2 * self.recall * self.precision / (self.recall + self.precision))
            if self.recall + self.precision > 0
            else 0.0
        )


@dataclass
class AccuracyResult:
    accuracy: float


@dataclass
class ICDARResult:
    per_doc_result: list[MetricResult]
    metrics: MetricResult = field(init=False)

    def __post_init__(self):
        recall = sum(r.recall for r in self.per_doc_result) / len(self.per_doc_result)
        precision = sum(r.precision for r in self.per_doc_result) / len(
            self.per_doc_result
        )
        self.metrics = MetricResult(recall, precision)


@dataclass
class LabelResult:
    per_doc_result: list[dict[label_config.TaskLabel, bool]]
    label_metrics: dict[label_config.TaskLabel, AccuracyResult] = field(init=False)

    def __post_init__(self):
        label_metrics = {}
        for lbl in label_config.TaskLabel:
            if lbl == label_config.TaskLabel.OTHER:
                continue
            label_metrics[lbl] = AccuracyResult(
                sum(r[lbl] for r in self.per_doc_result) / len(self.per_doc_result)
            )
        self.label_metrics = label_metrics


def compute_scores_icdar(tgt: list[str], pred: list[str]) -> MetricResult:
    """Mimics ICDAR SROIE scoring:
    * precision
    * recall
    * f1 score

    Args:
        tgt (list[str]): target values for each label, empty string if not target
        pred (list[str]): predicted value for each label, empty string if not prediction

    Returns:
        MetricResult: score of the prediction against targets
    """
    recall = (
        sum(t == p for t, p in zip(tgt, pred) if t) / sum(1 for t in tgt if t)
        if sum(1 for t in tgt if t) > 0
        else 0.0
    )
    precision = (
        sum(t == p for t, p in zip(tgt, pred) if p) / sum(1 for p in pred if p)
        if sum(1 for p in pred if p) > 0
        else 0.0
    )
    return MetricResult(recall, precision)


@dataclass
class FieldPrediction:
    value: Any = None
    positions: list[documents.Position] = field(default_factory=list)
    confidence: float = 0.
    parsed_value: Any = field(init=False)

    label: InitVar[label_config.TaskLabel] = label_config.TaskLabel.OTHER

    def __post_init__(self, label):
        parser = label.get_parser()
        self.parsed_value = parser(self.value)
        
    @property
    def confident_value(self):
        return self.value if self.confidence > CONFIDENCE_THRESHOLD_PRED else None
    
    @property
    def confident_parsed_value(self):
        return self.parsed_value if self.confidence > CONFIDENCE_THRESHOLD_PRED else None
        
    def export(self):
        return {
            "value": self.value,
            "parsed_value": self.parsed_value,
            
            "positions": [p.export() for p in self.positions],
            "confidence": self.confidence if isinstance(self.confidence, float) else self.confidence.item()
        }

@dataclass
class Prediction:
    """Handle for prediction result and metrics computation"""

    prediction: dict[label_config.TaskLabel, FieldPrediction] = field(
        default_factory=lambda: defaultdict(FieldPrediction)
    )

    def __getitem__(self, key):
        return self.prediction[key]

    def keys(self):
        return self.prediction.keys()

    def values(self):
        return self.prediction.values()

    def items(self):
        return self.prediction.items()

    def compare_icdar(self, target_file: Path) -> MetricResult:
        """Compare with icdar targets

        Args:
            target_file (Path): path to the json target file

        Returns:
            MetricResult: ICDAR scoring of the prediction according to the target
        """
        with target_file.open("r") as tgt_f:
            tgt = json.load(tgt_f)

        pred_values = []
        tgt_values = []

        for lbl in label_config.TaskLabel:
            if lbl == label_config.TaskLabel.OTHER:
                continue

            pred_values.append(self[lbl].confident_value)
            tgt_values.append(tgt.get(lbl.name.lower(), "") or "")

        return compute_scores_icdar(tgt_values, pred_values)

    def compare(self, target_file: Path) -> dict:
        """Compare with targets and score predicted classification

        Args:
            target_file (Path): path to the json target file

        Returns:
            dict: scoring of the prediction according to the target
        """
        with target_file.open("r") as tgt_f:
            tgt = json.load(tgt_f)

        result = {}

        for lbl in label_config.TaskLabel:
            if lbl == label_config.TaskLabel.OTHER:
                continue
            tgt_lbl = tgt.get(lbl.name.lower(), None) or None
            result[lbl] = self[lbl].confident_value == tgt_lbl

        return result
    
    def export(self) -> dict:
        res = {}
        solo_fields, tuple_fields = label_config.TaskLabel.export_fields()

        for field_name in solo_fields:
            lbl = label_config.TaskLabel.from_str(field_name)
            if lbl in self.prediction:
                res[field_name] = self[lbl].export()

        tuple_res = {}

        for i in range(1, 3):
            tuple_lbls = [label_config.TaskLabel.from_str(f"{field_name}_{i}") for field_name in tuple_fields]
            if all(
                lbl in self.prediction
                for lbl in tuple_lbls
            ):
                values = tuple([self[lbl].parsed_value for lbl in tuple_lbls])
                if all(values) and values not in tuple_res:
                    tuple_res[values] = [self[lbl].export() for lbl in tuple_lbls]
        if tuple_res:
            for field_name, values in zip(tuple_fields, zip(*tuple_res.values())):
                res[field_name] = values
        
        return res

    @staticmethod
    def merge_together(preds: list["Prediction"]) -> "Prediction":
        new_pred = defaultdict(FieldPrediction)
        for pred in preds:
            for lbl, value in pred.items():
                # assert lbl not in new_pred
                if lbl not in new_pred or new_pred[lbl].confidence < value.confidence:
                    new_pred[lbl] = value
        
        return Prediction(new_pred)


def get_word_idx_per_token_deprecated(
    offset_mapping: TensorType["sequence", 2, int]
) -> TensorType["sequence", int]:
    """Compute for each token which word index it corresponds to.

    Args:
        offset_mapping (TensorType): offset mapping computed by huggingface FastTokenizer (https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizerfast)

    Returns:
        TensorType: word index for each token in the input sequence
    """
    curr_word_idx = -1
    res = torch.ones(offset_mapping.size(0), dtype=int) * -1
    for i, (beg, end) in enumerate(offset_mapping):
        if beg == 0 and end > 0:
            # This token is the start of a new word
            curr_word_idx += 1
            res[i] = curr_word_idx
        elif beg > 0:
            # This token is the continuation of curr_word_idx
            res[i] = curr_word_idx
        # else:
        # Meta token -> no corresponding word

    return res

def get_token_mask_beg_of_word(
    offset_mapping: TensorType["sequence", 2, int]
) -> TensorType["sequence", int]:
    """Compute mask of tokens that correspond to the beginning of a word
    """
        
    start_mask = (offset_mapping[:, 0] == 0) & (offset_mapping[:, 1] != 0)
    if label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING:
        beg_context = next(i for i, offset in enumerate(offset_mapping[1:], 1) if tuple(offset) == (0, 0)) + 1
        start_mask[:beg_context] = False
        start_mask[0] = True  # First token is mandatory
    return start_mask


def run_model(
    model: transformers.PreTrainedModel,
    data: dataset_util.Encoding,
    output_attentions=False,
) -> TensorType["sequence", "model", float]:
    """Run a model on the data.
    data will be placed on model's device if devices mismatch

    Args:
        model (transformers.PreTrainedModel): ML pre trained model
        data (dataset_util.Encoding): data shard for one document
        output_attention (bool): if True, attention weights are return alongside logits

    Returns:
        TensorType: either logits or logits and attention weights
    """
    device = model.device
    output = model(
        **{k: v.unsqueeze(0).to(device) for k, v in data.items()},
        output_attentions=output_attentions,
    )
    logits: TensorType["sequence", "model", float] = output.logits.squeeze(0).detach()

    if output_attentions:
        attentions = output.attentions.squeeze(0).detach()
        return logits, attentions

    return logits

def get_mean_occurence_dp(
    prediction: TensorType["token_sequence", "classes", float],
    # label: label_config.TaskLabel,
    return_dp_value: bool = False,
) -> list[list[int]]:
    dp = [
        [(-math.inf, -1, 0) for _ in range(prediction.size(1))]
        for _ in range(prediction.size(0) + 1)
    ]

    max_dp_value = -math.inf
    max_dp_idx = -1

    for i in range(1, prediction.size(0) + 1):
        # B
        dp[i][0] = (prediction[i - 1, 0], -1, 1)

        # I
        dp[i][1] = (
            max(dp[i - 1][0][0], dp[i - 1][1][0])
            + (prediction[i - 1, 1] if prediction[i - 1, 1] > 0 else -5),
            0 if dp[i - 1][0][0] > dp[i - 1][1][0] else 1,
            1 + (dp[i-1][0][2] if dp[i-1][0][0] > dp[i-1][1][0] else dp[i-1][1][2])
        )

        # E
        dp[i][2] = (
            max(dp[i - 1][0][0], dp[i - 1][1][0])
            + (prediction[i - 1, 2] if prediction[i - 1, 2] > 0 else -5),
            0 if dp[i - 1][0][0] > dp[i - 1][1][0] else 1,
            1 + (dp[i-1][0][2] if dp[i-1][0][0] > dp[i-1][1][0] else dp[i-1][1][2])
        )

        # S
        dp[i][3] = (prediction[i - 1, 3], -1, 1)

        if any(dpi/n > max_dp_value for dpi, _, n in dp[i][2:4]):
            max_dp_value = max(dpi/n for dpi, _, n in dp[i][2:4])
            max_dp_idx = i

    if max_dp_value < CONFIDENCE_THRESHOLD_DP:
        if return_dp_value:
            return [], -math.inf
        return []

    offset = 2 if label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING else 1
    oc = [max_dp_idx - offset]  # Offset due to first token is META
    i = max_dp_idx
    dp_pred = max(dp[i])
    while dp_pred[1] >= 0:
        i -= 1
        dp_pred = dp[i][dp_pred[1]]
        oc.append(i - offset)  # Offset due to first token is META
    
    if return_dp_value:
        return [sorted(oc)], max_dp_value
    return [sorted(oc)]



def get_occurences_dp(
    prediction: TensorType["token_sequence", "classes", float],
    # label: label_config.TaskLabel,
    return_dp_value: bool = False,
) -> list[list[int]]:
    """Find the best occurence of label in the predictions by solving dynamic programming maximum confidence
    in the predictions.
    An occurence MUST start with tag B or S, end with E or S and I can only follow B or I and be followed by I or E.
    regex rule for occurence: BI*E|S

    Args:
        prediction (TensorType): logits predicted by model
        # label (documents.Label): instance of label to get occurences of

    Returns:
        list[list[int]]: Single element list containing a list of token indices composing the best occurence.
    """
    dp = [
        [(-math.inf, -1) for _ in range(prediction.size(1))]
        for _ in range(prediction.size(0) + 1)
    ]

    max_dp_value = -math.inf
    max_dp_idx = -1

    for i in range(1, prediction.size(0) + 1):
        # B
        dp[i][0] = (prediction[i - 1, 0], -1)

        # I
        dp[i][1] = (
            max(dp[i - 1][0][0], dp[i - 1][1][0])
            + (prediction[i - 1, 1] if prediction[i - 1, 1] > 0 else -5),
            0 if dp[i - 1][0][0] > dp[i - 1][1][0] else 1,
        )

        # E
        dp[i][2] = (
            max(dp[i - 1][0][0], dp[i - 1][1][0])
            + (prediction[i - 1, 2] if prediction[i - 1, 2] > 0 else -5),
            0 if dp[i - 1][0][0] > dp[i - 1][1][0] else 1,
        )

        # S
        dp[i][3] = (prediction[i - 1, 3], -1)

        if any(dpi > (max_dp_value,) for dpi in dp[i][2:4]):
            max_dp_value = max(dp[i])[0]
            max_dp_idx = i

    if max_dp_value < CONFIDENCE_THRESHOLD_DP:
        if return_dp_value:
            return [], -math.inf
        return []

    offset = 2 if label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING else 1
    oc = [max_dp_idx - offset]  # Offset due to first token is META
    i = max_dp_idx
    dp_pred = max(dp[i])
    while dp_pred[1] >= 0:
        i -= 1
        dp_pred = dp[i][dp_pred[1]]
        oc.append(i - offset)  # Offset due to first token is META
    
    if return_dp_value:
        return [sorted(oc)], max_dp_value
    return [sorted(oc)]


def get_occurences(
    prediction: TensorType["token_sequence", int],
    label: label_config.TaskLabel,
    *,
    offset: int = 1,
) -> list[list[int]]:
    """Find any occurences of label using strict rule interpretation of BIESO tags.
    Occurence starts with B or S, B can be followed by I or E. E and S ends the current occurence.
    regex rule for occurence: BI*E|S

    Args:
        prediction (TensorType): predicted labels by model
        label (documents.Label): instance of label to get occurences of
        offset (int, optional): Offset to apply at the beginning of the sequence to keep token indices correct. Defaults to 1.

    Returns:
        list[list[int]]: all found occurences. Each occurence is a list containing its token indices.
    """

    occurences = []
    for i, val in enumerate(prediction):
        pred_tag = dataset_util.BIESOTag.from_int(val.item())
        if pred_tag.label == label:
            if pred_tag.tag == "B":
                occurences.append([i + offset])
            elif pred_tag.tag == "S":
                occurences.append([i + offset])
            elif pred_tag.tag in "IE":
                if len(occurences) > 0 and len(occurences[-1]) > 0:
                    occurences[-1].append(i + offset)

            if pred_tag.tag in "SE":
                occurences.append([])

    return [oc for oc in occurences if len(oc) > 0]


def _predict_one_seq_tagging(
    model: transformers.PreTrainedModel,
    data: dataset_util.Encoding,
    doc: documents.LabeledDocument,
    use_dp: bool = True,
) -> Prediction:
    """Compute model's predictions for input document

    Args:
        model (transformers.PreTrainedModel): pre trained model that classify each token into Tag
        data (dataset_util.Encoding): encoded data shard
        document (document.LabeledDocument): labeled document encoded data comes from. Labels are not used

    Returns:
        Prediction:
    """
    logits = run_model(model, data).cpu()
    pred_cls = logits  # .argmax(dim=-1).long()

    # word_indices = get_word_idx_per_token_deprecated(data.offset_mapping)
    # last_token = (
    #     next(i for i, v in reversed(list(enumerate(word_indices))) if v != -1) + 1
    # )
    
    begin_word_indices = get_token_mask_beg_of_word(data.offset_mapping)
    prediction = defaultdict(FieldPrediction)

    for lbl in label_config.TaskLabel:
        if lbl == label_config.TaskLabel.OTHER:
            continue
        beg_cls, end_cls = int(dataset_util.BIESOTag(lbl, "B")), int(
            dataset_util.BIESOTag(lbl, "S")
        )
        if use_dp:
            occurences_idx, confidence_value = get_occurences_dp(pred_cls[begin_word_indices, beg_cls : end_cls + 1], return_dp_value=True)
        else:
            occurences_idx = get_occurences(pred_cls[begin_word_indices, :].argmax(-1), lbl, offset=0)
            confidence_value = 10.

        occurence_number = 0

        if len(occurences_idx) > 0:
            token_indices = occurences_idx[occurence_number]
            # lbl_word_indices = torch.unique(word_indices[token_indices], sorted=True)
            prediction_value = " ".join(
                doc.document.words[i]
                for i in token_indices
                if len(doc.document.words) > i >= 0
            )
            prediction_positions = [doc.document.positions[i].invert_normalization(*doc.document.doc_size) for i in token_indices if len(doc.document.words) > i >= 0]
            prediction[lbl] = FieldPrediction(prediction_value, prediction_positions, confidence=confidence_value, label=lbl)

    return Prediction(prediction)


def _predict_one_span_qa(
    model: transformers.PreTrainedModel,
    data: dataset_util.SpanQAEncoding,
    doc: documents.LabeledDocument,
) -> Prediction:
    lbl = data.qa_label
    logits = run_model(model, data).cpu()
    # TODO: Sample according to confidence instead of argmax
    pred_cls = logits  # .argmax(dim=-1).long()

    begin_word_indices = get_token_mask_beg_of_word(data.offset_mapping)
    prediction = defaultdict(FieldPrediction)

    occurences_idx, confidence_value = get_occurences_dp(pred_cls[begin_word_indices, 1 : ], return_dp_value=True)
    occurence_number = 0

    # print()
    # print(begin_word_indices[:10])
    # print(len(doc.document.words))
    # print(occurences_idx)

    if len(occurences_idx) > occurence_number and not -1 in occurences_idx[occurence_number]:
        token_indices = occurences_idx[occurence_number]
        # lbl_word_indices = torch.unique(word_indices[token_indices], sorted=True)
        prediction_value = " ".join(
            doc.document.words[i]
            for i in token_indices
            if len(doc.document.words) > i >= 0
        )
        prediction_positions = [doc.document.positions[i].invert_normalization(*doc.document.doc_size) for i in token_indices]
        prediction[lbl] = FieldPrediction(prediction_value, prediction_positions, confidence=confidence_value, label=lbl)

    return Prediction(prediction)

def predict_one(
    model: transformers.PreTrainedModel,
    data: dataset_util.Encoding,
    doc: documents.LabeledDocument,
    use_dp: bool = True,
) -> Prediction:
    if label_config.SelectedTask == label_config.TaskType.SEQUENCE_TAGGING:
        return _predict_one_seq_tagging(model, data, doc, use_dp)
    elif label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING:
        return _predict_one_span_qa(model, data, doc)
    else:
        raise NotImplementedError()


def get_truth(
    doc: documents.LabeledDocument
):
    prediction = defaultdict(FieldPrediction)
    for lbl in label_config.TaskLabel:
        if lbl == label_config.TaskLabel.OTHER:
            continue
        
        prediction_value = " ".join(w for w, l in zip(doc.document.words, doc.labels) if l == lbl)
        prediction_positions = [b.invert_normalization(*doc.document.doc_size) for b, l in zip(doc.document.positions, doc.labels) if l == lbl]
        
        prediction[lbl] = FieldPrediction(prediction_value, prediction_positions, confidence=math.inf, label=lbl)
    return Prediction(prediction)


def evaluate(
    model: transformers.PreTrainedModel, dataset: dataset_util.DocumentDataset, use_dp: bool = True
):
    """Evaluates model on dataset

    Args:
        model (transformers.PreTrainedModel): pre trained model that classify each token into Tag
        dataset (dataset_util.DocumentDataset): dataset containing documents

    Returns:
        [type]: icdar sroie style results, per class results and individual predictions
    """
    model.eval()
    doc_d = {(doc.document.filename, doc.part): doc for doc in dataset.documents}

    all_icdar_results = []
    all_label_results = []
    all_predictions = defaultdict(list)

    for data in tqdm(dataset, total=len(dataset)):
        file = data.filename
        part = data.part
        doc = doc_d[(file, part)]
        pred = predict_one(model, data, doc, use_dp)

        all_predictions[file].append(pred)

    final_predictions = []
    for file, preds in all_predictions.items():
        pred = Prediction.merge_together(preds)
        final_predictions.append(pred)

        if label_config.conf["LABEL"]["Name"] == "ICDAR":
            result_icdar = pred.compare_icdar(Path("data/icdar/test/tgt") / file.name)
            result_label = pred.compare(Path("data/icdar/test/tgt") / file.name)
        elif label_config.conf["LABEL"]["Name"] == "BDCPO":
            result_icdar = pred.compare_icdar(Path("/data/bdcpo/post_processed_predictions") / file.name.replace("jpg", "txt"))
            result_label = pred.compare(Path("/data/bdcpo/post_processed_predictions") / file.name.replace("jpg", "txt"))

        all_icdar_results.append(result_icdar)
        all_label_results.append(result_label)

    icdar_result = ICDARResult(all_icdar_results)
    label_result = LabelResult(all_label_results)

    return icdar_result, label_result, final_predictions


def main():
    from src.train import set_all_seeds, get_model
    from src import utils, icdar_sroie

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)

    args = parser.parse_args()
    set_all_seeds(1)


    # model = get_model(model_name=model_name, local_attention_threshold=local_attention_threshold)
    model = get_model(saved_model=args.folder)

    if label_config.conf["LABEL"]["Name"] == "ICDAR":
        test_dataset = dataset_util.DocumentDataset(
            icdar_sroie.get_docs_from_disk(Path("data/icdar/test"))
        )
    elif label_config.conf["LABEL"]["Name"] == "BDCPO":
        test_dataset = dataset_util.DocumentDataset(documents.load_from_layoutlm_style_dataset(Path("/data/bdcpo"), "test"))
    else:
        raise Exception("Not available")

    doc_l = test_dataset.documents
    for data, doc in zip(test_dataset, doc_l):
        pred = predict_one(model, data, doc)
        print(pred)
        print(pred.export())
        input()
    # with open(Path(args.folder) / "results.json", "w") as fp:
    #     json.dump(results, fp, cls=utils.DataclassJSONEncoder)


if __name__ == "__main__":
    main()
