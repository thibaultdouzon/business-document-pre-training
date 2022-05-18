import collections
import json
import os
import random
import time

from pathlib import Path
from typing import Optional

import more_itertools
import numpy as np
import torch
import transformers

from transformers import LayoutLMForMaskedLM, TrainingArguments, Trainer, PreTrainedModel
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src import train, dataset_pretrain, dataset_util
from src.modeling.layoutlm_pre.modeling_layoutlm_pre import LayoutLMPreTrainForMultiTask

PRETRAIN_PATH = Path(R"\\ly-ml\D\IEasQuestionAnswering\pretrain_dataset") if os.name == "nt" else Path(R"/data/pretrain_dataset")
# PRETRAIN_PATH = Path(R"data/pretrain")

def get_pretrain_model(saved_model: Optional[Path] = None, *, model_name: str = "layoutlm", multitask=False, **kwargs) -> PreTrainedModel:
    """Fetch a model from local sources or pretrained online ressources

    Returns:
        LayoutLMForTokenClassification: model with loaded weights
    """
    if model_name == "layoutlm":
        cls = LayoutLMPreTrainForMultiTask if multitask else LayoutLMForMaskedLM
    else:
        raise NotImplementedError

    if saved_model is None:
        return cls.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=dataset_util.DEFAULT_TOKENIZER.vocab_size,
            **kwargs,
        )
    else:
        return cls.from_pretrained(saved_model, **kwargs)


def get_pretrain_dataset() -> tuple[dataset_pretrain.PreTrainDocumentDatasetMLM, dataset_pretrain.PreTrainDocumentDatasetMLM]:
    """Fetch dataset and prepare data for use with huggingface models

    Returns:
        tuple[dataset_util.DocumentDataset, dataset_util.DocumentDataset]: train and eval datasets
    """

    train_dataset = dataset_pretrain.PreTrainDocumentDatasetMLM(list(PRETRAIN_PATH.glob("train_ap/*")) +
                                                             list(PRETRAIN_PATH.glob("train_om/*"))
                                                             )
    eval_dataset = dataset_pretrain.PreTrainDocumentDatasetMLM(list(PRETRAIN_PATH.glob("test_ap/*")) +
                                                            list(PRETRAIN_PATH.glob("test_om/*"))
                                                            )

    return train_dataset, eval_dataset

def get_multipretrain_dataset() -> tuple[ConcatDataset, ConcatDataset]:
    """Fetch dataset and prepare data for use with huggingface models

    Returns:
        tuple[dataset_util.DocumentDataset, dataset_util.DocumentDataset]: train and eval datasets
    """

    train_dataset_mlm = dataset_pretrain.PreTrainDocumentDatasetMLM(list(PRETRAIN_PATH.glob("train_ap/*")) +
                                                                    list(PRETRAIN_PATH.glob("train_om/*"))
                                                                   )
    train_dataset_num = dataset_pretrain.PreTrainDocumentDatasetNumber(list(PRETRAIN_PATH.glob("train_ap/*")) +
                                                                    list(PRETRAIN_PATH.glob("train_om/*"))
                                                                   )
    train_dataset_lay = dataset_pretrain.PreTrainDocumentDatasetLayout(list(PRETRAIN_PATH.glob("train_ap/*")) +
                                                                    list(PRETRAIN_PATH.glob("train_om/*"))
                                                                   )
    eval_dataset_mlm = dataset_pretrain.PreTrainDocumentDatasetMLM(list(PRETRAIN_PATH.glob("test_ap/*")) +
                                                                   list(PRETRAIN_PATH.glob("test_om/*"))
                                                                  )
    eval_dataset_num = dataset_pretrain.PreTrainDocumentDatasetNumber(list(PRETRAIN_PATH.glob("test_ap/*")) +
                                                                   list(PRETRAIN_PATH.glob("test_om/*"))
                                                                  )
    eval_dataset_lay = dataset_pretrain.PreTrainDocumentDatasetLayout(list(PRETRAIN_PATH.glob("test_ap/*")) +
                                                                   list(PRETRAIN_PATH.glob("test_om/*"))
                                                                  )

    return ConcatDataset([train_dataset_mlm, train_dataset_num, train_dataset_lay]), ConcatDataset([eval_dataset_mlm, eval_dataset_num, eval_dataset_lay])


class MultiObjectiveDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset: torch.utils.data.ConcatDataset,
                 samplers: list[torch.utils.data.sampler.Sampler],
                 **kwargs):
        self.dataset = dataset
        
        # transformers API expects sampler attribute
        self.sampler = samplers[0]
        
        self._dataloaders = [
            torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=sampler,
                **kwargs
                # batch_size=self.args.train_batch_size,
                # collate_fn=self.data_collator,
                # drop_last=self.args.dataloader_drop_last,
                # num_workers=self.args.dataloader_num_workers,
                # pin_memory=self.args.dataloader_pin_memory,
            )
            for dataset, sampler in zip(self.dataset, samplers)
        ]
    
    def __len__(self):
        return sum(map(len, self._dataloaders))
    
    def __iter__(self):
        yield from more_itertools.roundrobin(*self._dataloaders)


class PreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_train_sampler(self) -> Optional[list[torch.utils.data.sampler.Sampler]]:
        if any(isinstance(ds, torch.utils.data.IterableDataset) or not isinstance(
            ds, collections.abc.Sized
        ) for ds in self.train_dataset.datasets):
            return None
        
        if self.args.world_size <= 1:
                return [torch.utils.data.sampler.RandomSampler(ds) for ds in self.train_dataset.datasets]
        elif (
            self.args.parallel_mode in [transformers.training_args.ParallelMode.TPU, transformers.training_args.ParallelMode.SAGEMAKER_MODEL_PARALLEL]
            and not self.args.dataloader_drop_last
        ):
            raise NotImplementedError("Sampler not working on TPU")
        else:
            return [torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=self.args.world_size, rank=self.args.process_index
            ) for ds in self.train_dataset.datasets]

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_samplers = self._get_train_sampler()
        
        if train_samplers is None:
            raise ValueError("Trainer: training requires a sampler")
    
        assert isinstance(self.train_dataset, torch.utils.data.ConcatDataset)
        
        return MultiObjectiveDataLoader(
            dataset=self.train_dataset.datasets,
            samplers=train_samplers,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("-m", "--multitask", action="store_true")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    
    train.set_all_seeds(42)
    
    train_dataset, eval_dataset = get_multipretrain_dataset() if args.multitask else get_pretrain_dataset()
    print(len(train_dataset))
    
    save_dir = "test_pretrain"
    model_name = "layoutlm"

    model = get_pretrain_model(model_name=model_name, multitask=args.multitask)
        
    training_args = TrainingArguments(
        output_dir="./results_pretrain",  # output directory
        num_train_epochs=5,  # total number of training epochs
        logging_dir=f"./logs/{model_name}_{int(time.time())}",  # directory for storing logs
        # max_steps=100_000,
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        gradient_accumulation_steps=8,  # number of gradient accumulation steps during training
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.005,  # strength of weight decay
        logging_steps=10,
        logging_strategy="steps",
        eval_steps=500,
        evaluation_strategy="steps",
        report_to="tensorboard",
        fp16=True,
        local_rank=args.local_rank if args.parallel else -1
        # no_cuda=True,
    )
    if args.multitask:
        trainer = PreTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=eval_dataset,  # evaluation dataset
            # compute_metrics=model_metrics,
        )
    else:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=eval_dataset,  # evaluation dataset
            # compute_metrics=model_metrics,
        )

    trainer.train()
    train.save_model(model, save_dir)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
