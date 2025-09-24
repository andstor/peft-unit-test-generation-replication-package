#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


import sys
import logging
import math
import os
import sys
import warnings

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


from peft import get_peft_model

# Local imports
from utils import WandbPredictionProgressCallback
from arguments import ModelArguments, DataTrainingArguments, LoggingArguments, PeftArguments


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.41.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((PeftArguments, ModelArguments, DataTrainingArguments, LoggingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        peft_args, model_args, data_args, logging_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        peft_args, model_args, data_args, logging_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

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
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.

        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,

    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(f"Tokenizer does not have a pad token, setting it to `{tokenizer.eos_token_id=}`.")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
        # this is here to save gpu vram. Likely only needed when using 40b or when oom issues happen ref: https://stackoverflow.com/questions/76633335/why-does-hugging-face-falcon-model-use-mode-config-use-cache-false-why-wouldn
        #model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # Initialize PEFT model if needed
    if peft_args.use_lora or peft_args.use_prompt_tuning or peft_args.use_ia3:
        
        if peft_args.use_prompt_tuning:
            from peft import PromptTuningConfig, PromptTuningInit, TaskType
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=peft_args.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.TEXT if peft_args.virtual_tokens_init_text else PromptTuningInit.RANDOM,
                prompt_tuning_init_text=peft_args.virtual_tokens_init_text,
                tokenizer_name_or_path=tokenizer.init_kwargs["name_or_path"],
            )

        if peft_args.use_lora:
            from peft import LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=peft_args.rank,
                target_modules=peft_args.target_modules,
                lora_alpha=peft_args.lora_alpha,
                lora_dropout=peft_args.lora_dropout,
                fan_in_fan_out=peft_args.fan_in_fan_out,
                bias=peft_args.bias,
                modules_to_save=peft_args.modules_to_save,
                layers_to_transform=peft_args.layers_to_transform,
                layers_pattern=peft_args.layers_pattern,
                rank_pattern=peft_args.rank_pattern,
                alpha_pattern=peft_args.alpha_pattern
            )

        if peft_args.use_ia3:
            from peft import IA3Config, TaskType
            peft_config = IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=peft_args.target_modules,
                feedforward_modules=peft_args.feedforward_modules,
                fan_in_fan_out=peft_args.fan_in_fan_out,
                modules_to_save=peft_args.modules_to_save,
                init_ia3_weights=peft_args.init_ia3_weights
            )
            

        model = get_peft_model(model, peft_config, adapter_name=peft_args.adapter_name)


    # Preprocessing the datasets.

    # Data collator
    #label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        #label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) seems to be larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={data_args.block_size}."
            )
        # Some models have inproperly tokenizer.model_max_length, so we allow overriding it
        #block_size = min(data_args.block_size, tokenizer.model_max_length)
        block_size = data_args.block_size
    
    if peft_args.use_prompt_tuning:
        block_size -= peft_args.num_virtual_tokens
        if data_collator.pad_to_multiple_of is not None:
            block_size -= data_collator.pad_to_multiple_of - (peft_args.num_virtual_tokens % data_collator.pad_to_multiple_of)
            logger.warning(f"Prompt tuning is used while inputs are padded to multiple of {data_collator.pad_to_multiple_of}. "
                        f"New block size is {block_size} to accomodate virtual tokens.")


    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            input_ids = None
            attention_mask = None
            for col in data_args.text_column_names:
                if input_ids is None:
                    tokenized_example = tokenizer(examples[col], add_special_tokens=True)
                    input_ids = tokenized_example["input_ids"]
                    attention_mask = tokenized_example["attention_mask"]
                else:
                    tokenized_example = tokenizer(examples[col], add_special_tokens=False)
                    [input_ids[i].extend(tokenized_example["input_ids"][i]) for i in range(len(input_ids))]
                    [attention_mask[i].extend(tokenized_example["attention_mask"][i]) for i in range(len(attention_mask))]
                
            if data_args.target_colum_name is not None:
                tokenized_example = tokenizer(examples[data_args.target_colum_name], add_special_tokens=False)
                labels = [[-100]*len(input_ids[i]) for i in range(len(input_ids))]
                [input_ids[i].extend(tokenized_example["input_ids"][i]) for i in range(len(input_ids))]
                [attention_mask[i].extend(tokenized_example["attention_mask"][i]) for i in range(len(attention_mask))]
                [labels[i].extend(tokenized_example["input_ids"][i]) for i in range(len(labels))]
            else:
                labels = input_ids.copy()

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - too long input will be filtered out"
                " before being passed to the model."
            )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def filter_toolong(examples):
        return  [ len(ex) <= block_size for ex in examples["input_ids"] ]


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        column_names = list(train_dataset.features)
        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on dataset")
                train_dataset = train_dataset.filter(filter_toolong, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache, desc="Filtering toolong")
            else:
                train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=column_names)
                train_dataset = train_dataset.filter(filter_toolong, batched=True)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))


    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        column_names = list(eval_dataset.features)
        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on dataset")
                eval_dataset = eval_dataset.filter(filter_toolong, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache, desc="Filtering toolong")
            else:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=column_names)
                eval_dataset = eval_dataset.filter(filter_toolong, batched=True)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))


    # Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:]
        preds = preds[:, :-1]
        if peft_args.use_prompt_tuning:
            preds = preds[:, peft_args.num_virtual_tokens:]
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        return metric.compute(predictions=preds, references=labels)


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


    training_args.gradient_checkpointing_kwargs={"use_reentrant": False} if training_args.gradient_checkpointing else None
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )

    if logging_args.log_preditions:
        progress_callback = WandbPredictionProgressCallback(trainer, tokenizer, eval_dataset, logging_args.log_predition_samples)
        trainer.add_callback(progress_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        ### if deepspeed is used, and half precision is used, to save the model in fp32 we need a checkpoint that we can extract the fp32 model from.
        ### see https://github.com/huggingface/transformers/issues/28921
        if training_args.deepspeed and (training_args.fp16 or training_args.bf16):
            logger.warning(
                "Saved model is in half-precision. "
                "Saving extra checkpoint for allowing fp32 parameters extraction. "
                "Use the zero_to_fp32.py script to extract the fp32 model from the checkpoint.")
            trainer.model_wrapped.save_checkpoint(training_args.output_dir)
        ###

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

    