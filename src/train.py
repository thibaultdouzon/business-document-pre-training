import argparse
import json
import os
import random
import time

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers

from transformers import LayoutLMForTokenClassification, TrainingArguments, Trainer, PreTrainedModel
from icecream import install
from sklearn.metrics import precision_recall_fscore_support

install()

from src import dataset_util, icdar_sroie, prediction, utils, label_config, documents
from src.modeling.layoutlm_crf.modeling_layoutlm_crf import LayoutLMCRFForTokenClassification



def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_model(saved_model: Optional[Path] = None, *, model_name: str = "layoutlm", **kwargs) -> PreTrainedModel:
    """Fetch a model from local sources or pretrained online ressources

    Returns:
        LayoutLMForTokenClassification: model with loaded weights
    """
    if model_name == "layoutlm":
        cls = LayoutLMForTokenClassification
    elif model_name == "layoutlm_crf":
        cls = LayoutLMCRFForTokenClassification
    else:
        raise NotImplementedError

    if saved_model is None:
        return cls.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=dataset_util.NUM_OUT_TAGS,
            **kwargs,
        )
    else:
        return cls.from_pretrained(saved_model, **kwargs)

def get_dataset(ignore_non_beg=True) -> tuple[dataset_util.DocumentDataset, dataset_util.DocumentDataset]:
    """Fetch dataset and prepare data for use with huggingface models

    Returns:
        tuple[dataset_util.DocumentDataset, dataset_util.DocumentDataset]: train and eval datasets
    """
    if label_config.conf["LABEL"]["Name"] == "ICDAR":
        # all_docs = icdar_sroie.get_docs_from_disk(Path("data/icdar/train"))
        # random.shuffle(all_docs)
        # train_docs, eval_docs = all_docs[:-20], all_docs[-20:]
        train_docs = documents.load_from_layoutlm_style_dataset(Path("data/SROIE2019"), "train")
        eval_docs = documents.load_from_layoutlm_style_dataset(Path("data/SROIE2019"), "dev")

    elif label_config.conf["LABEL"]["Name"] == "BDCPO":
        train_docs = documents.load_from_layoutlm_style_dataset(Path("/data/bdcpo"), "train")
        eval_docs = documents.load_from_layoutlm_style_dataset(Path("/data/bdcpo"), "dev")

    train_dataset = dataset_util.DocumentDataset(train_docs, ignore_non_beg_token=ignore_non_beg)
    eval_dataset = dataset_util.DocumentDataset(eval_docs, ignore_non_beg_token=ignore_non_beg)

    return train_dataset, eval_dataset


def save_model(model: transformers.PreTrainedModel, name: str):
    model.save_pretrained(Path("saved_models") / name)


def model_metrics(eval_pred: transformers.EvalPrediction):
    id_mask = eval_pred.label_ids >= 0
    
    precision, recall, f1, *_ = precision_recall_fscore_support(eval_pred.label_ids[id_mask].reshape(-1),
        eval_pred.predictions.argmax(-1)[id_mask].reshape(-1),
        labels=list(range(dataset_util.NUM_OUT_TAGS)),
        zero_division=0)

    if label_config.SelectedTask == label_config.TaskType.SEQUENCE_TAGGING:
        return {
            **{f"precision_{dataset_util.BIESOTag.from_int(i).label.name}_{dataset_util.BIESOTag.from_int(i).tag}": p for i, p in enumerate(precision)},
            **{f"recall_{dataset_util.BIESOTag.from_int(i).label.name}_{dataset_util.BIESOTag.from_int(i).tag}": p for i, p in enumerate(recall)},
            **{f"f1_{dataset_util.BIESOTag.from_int(i).label.name}_{dataset_util.BIESOTag.from_int(i).tag}": p for i, p in enumerate(f1)}
            }
    elif label_config.SelectedTask == label_config.TaskType.SPAN_QUESTION_ANSWERING:
        return {
            **{f"precision_{dataset_util.SpanTag(i)!s}": p for i, p in enumerate(precision)},
            **{f"recall_{dataset_util.SpanTag(i)!s}": p for i, p in enumerate(recall)},
            **{f"f1_{dataset_util.SpanTag(i)!s}": p for i, p in enumerate(f1)}
            }


def train(seed, save_dir, n_epochs=-1, n_steps=-1, **kwargs):
    """Train model according to args"""
    
    set_all_seeds(seed)
    
    save_dir = f"{save_dir}_{seed}"
    model_name = "layoutlm_crf"
    use_dp = False
    local_attention_threshold = 200

    # state = torch.load("./docker/pretrain/saved_models/test_pretrain/pytorch_model.bin")
    
    train_dataset, eval_dataset = get_dataset()

    # model = get_model(model_name=model_name, local_attention_threshold=local_attention_threshold)
    # model = get_model(model_name=model_name, state_dict=state)
    model = get_model(model_name=model_name)

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=n_epochs,  # total number of training epochs
        logging_dir=f"./logs/{model_name}_{int(time.time())}",  # directory for storing logs
        max_steps=n_steps,
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        gradient_accumulation_steps=6,  # number of gradient accumulation steps during training
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.008,  # strength of weight decay
        logging_steps=10,
        logging_strategy="steps",
        eval_steps=100,
        evaluation_strategy="steps",
        report_to="tensorboard",
        fp16=True,
        # no_cuda=True,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=model_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    save_model(model, save_dir)

    if label_config.conf["LABEL"]["Name"] ==  "ICDAR":
        test_dataset = dataset_util.DocumentDataset(icdar_sroie.get_docs_from_disk(Path("data/icdar/test")))
    elif label_config.conf["LABEL"]["Name"] == "BDCPO":
        test_dataset = dataset_util.DocumentDataset(documents.load_from_layoutlm_style_dataset(Path("/data/bdcpo"), "test"))

    *results, preds = prediction.evaluate(model, test_dataset, use_dp)
    with open(Path("saved_models") / save_dir / "results.json", "w") as fp:
        json.dump(results, fp, cls=utils.DataclassJSONEncoder)
    with open(Path("saved_models") / save_dir / "predictions.json", "w") as fp:
        json.dump([pred.export() for pred in preds], fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",)
    parser.add_argument("--num_runs", type=int)

    train_length = parser.add_mutually_exclusive_group()
    train_length.add_argument("--n_epochs", type=int, default=-1)
    train_length.add_argument("--n_steps", type=int, default=-1)

    args = parser.parse_args()

    for i in range(args.num_runs):
        train(seed=i, **vars(args))
    


if __name__ == "__main__":
    main()
