
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['http_proxy'] = '10.40.50.225:7890'
os.environ['https_proxy'] = '10.40.50.225:7890'


import torch

# print using GPU Info
if abs(int(os.environ.get('LOCAL_RANK', -1))) == 1:
    print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}.")

    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

import logging

import sys
from dataclasses import dataclass, field

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    LlamaModel,
    LlamaForCausalLM,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM, TaskType
from fastchat.model.model_adapter import get_conversation_template

from utils import load_dataset_from_json
from module.DatasetModule import DatasetModule
from module.trainer import NLUTrainer
from module.ModelModule import LlamaForNLU

from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer


# torch.cuda.set_device(5)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    num_labels: int = field(
        metadata={"help": "The number of labels in NLU task."}
    )


    llama_based: bool = field(
        metadata={"help": "Wheather the model is a llama based model."}
    )
    
    require_template: bool = field(
        metadata={"help": "Wheather the model is trained under specific instruction."}
    )
    require_lora: bool = field(
        metadata={"help": "Wheather the lora is required."}
    )
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    lora_rank: Optional[int] = field(
        default=None, metadata={"help": "The rank of LoRa"})
    
    lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "The scaling coefficient of LoRa"})
    
    lora_dropout: Optional[int] = field(
        default=None, metadata={"help": "The dropout rate of LoRa"})
    
    lora_target_modules: Optional[str] = field(
        default=None, metadata={"help": "The target modules of LoRa, e.g. ['k_proj', ..]"})
    
    lora_bias: Optional[str] = field(
        default="none", metadata={"help": "Whether LoRa require biases"})
    
    lora_config: Optional[str] = field(
        default=None, metadata={"help": "LoRa config path."}
    )
    
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    task: str = field(
        metadata={"help": "Which task in process."}
    )
    
    require_inv: str = field(
        metadata={"help": "Wheather inv input is required."}
    )
    
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    input_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

data_name_mapping = {
    "imdb": ("text", "label"),
    "conll2003": ("tokens", "ner_tags")
}

conll_ner_tags_mapping = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
conll_ner_tags_mapping = {v: k for k, v in conll_ner_tags_mapping.items()}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    sys.argv.append('/data/gj/Bi-Attention-HFTrainer/config/config.json')
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    
    logger.info("Bi-Attention employed: {}".format(str(data_args.require_inv)))
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    if data_args.train_file is not None:
        train_dataset = load_dataset_from_json(data_args.train_file)
    if data_args.validation_file is not None:
        val_dataset = load_dataset_from_json(data_args.validation_file)
    if data_args.test_file is not None:
        test_dataset = load_dataset_from_json(data_args.test_file)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if model_args.llama_based:            
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        
        model = LlamaModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16
        )
        
        for p in model.parameters():
            p.requires_grad = False
                
        # elif model_args.lora_config:
        #     model = AutoPeftModelForCausalLM.from_pretrained(lora_config)
    else:
        raise NotImplementedError

    
    if model_args.require_lora and model_args.lora_rank and model_args.lora_alpha and  model_args.lora_dropout:
        if training_args.do_train:
            lora_config = LoraConfig(TaskType.CAUSAL_LM, r=model_args.lora_rank,
                                    lora_alpha=model_args.lora_alpha,
                                    target_modules=model_args.lora_target_modules, 
                                    bias=model_args.lora_bias,
                                    lora_dropout=model_args.lora_dropout)
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
    
    model = LlamaForNLU(model, model_args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if data_args.train_file is None:
            raise ValueError("--do_train requires a train dataset")
    elif training_args.do_eval:
        if data_args.validation_file is None:
            raise ValueError("--do_eval requires a validation dataset")
    elif training_args.do_predict:
        if data_args.test_file is None:
            raise ValueError("--do_predict requires a test dataset")
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = data_name_mapping.get(data_args.dataset_name, None)
    
    if data_args.input_column is None:
        input_column = dataset_columns[0]
    else:
        input_column = data_args.input_column
        if input_column not in dataset_columns:
            raise ValueError(
                f"--text_column' value '{data_args.input_column}' needs to be one of: {', '.join(dataset_columns)}"
            )
    if data_args.target_column is None:
        target_column = dataset_columns[1]
    else:
        target_column = data_args.target_column
        if target_column not in dataset_columns:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(dataset_columns)}"
            )

    system_message = ''
    if model_args.require_template and model_args.llama_based:
        conv = get_conversation_template(model_args.config_name if model_args.config_name else model_args.model_name_or_path)
        roles = conv.roles
        system_message = conv.system_message
        

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset[ :max_train_samples]
        train_dataset = DatasetModule(train_dataset, system_message=system_message, 
                                      roles=roles, prefix=prefix, tokenizer=tokenizer,
                                      dataset_columns=dataset_columns, data_args = data_args)
        

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(val_dataset), data_args.max_eval_samples)
            val_dataset = val_dataset[ :max_eval_samples]
        val_dataset = DatasetModule(val_dataset, system_message=system_message, 
                                      roles=roles, prefix=prefix, tokenizer=tokenizer,
                                      dataset_columns=dataset_columns, data_args = data_args)

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset[ :max_predict_samples]
        test_dataset = DatasetModule(test_dataset, system_message=system_message, 
                                      roles=roles, prefix=prefix, tokenizer=tokenizer,
                                      dataset_columns=dataset_columns, data_args = data_args)

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("seqeval")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def postprocess_ner_ids(preds, labels):
        map_func = np.vectorize(lambda num: conll_ner_tags_mapping.get(num, ''))
        preds = np.where(labels != -100, preds, np.full_like(preds, -100))
        preds = map_func(preds).tolist()
        labels = map_func(labels).tolist()
        
        for pred in preds:
            pred[:] = [p for p in pred if p != '']
            
        for label in labels:
            label[:] = [l for l in label if l != '']
        return preds, labels 
        
    def compute_rouges(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    def compute_seqeval(eval_preds):
        def flatten_dict(d, parent_key='', sep='/'):
            items = {}
            for key, value in d.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if isinstance(value, dict):
                    items.update(flatten_dict(value, new_key, sep=sep))
                else:
                    items[new_key] = value
            return items

        
        def multiply(result, times):
            for k, v in result.items():
                if isinstance(v, dict):
                    result[k] = multiply(v, times)
                else:
                    result[k] = round(v * 100, 4)
            return result
        preds, labels = eval_preds
            
        # Some simple post-processing
        preds, labels = postprocess_ner_ids(preds, labels)

        result = metric.compute(predictions=preds, references=labels)
        result = multiply(result, 100)

        return flatten_dict(result)

    def compute_acc_pre_recall(eval_preds):
        preds, labels = eval_preds
        accuracy_result = accuracy.compute(predictions=preds, references=labels)
        recall_res = recall.compute(predictions=preds, references=labels)
        precision_res = precision.compute(predictions=preds, references=labels)
        
        result = {k: v for dic in [accuracy_result, recall_res, precision_res] for k, v in dic.items()}
        return result
        
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize our Trainer
    # optimizer = ZeroRedundancyOptimizer(filter(lambda p: p.requires_grad==True , model.parameters()),
    #                                 optimizer_class = AdamW,
    #                                 lr = training_args.learning_rate)
    optimizer = AdamW(filter(lambda p: p.requires_grad==True , model.parameters()), lr=training_args.learning_rate)
    trainer = NLUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_seqeval if data_args.task == 'ner 'else compute_acc_pre_recall,
        optimizers=(optimizer, None)
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(val_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in val_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train == False and training_args.resume_from_checkpoint is not None:
        trainer.load_cls_head_from_checkpoint(training_args.resume_from_checkpoint)
        
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()