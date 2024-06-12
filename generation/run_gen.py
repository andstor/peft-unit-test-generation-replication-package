import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig, AutoConfig
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteriaList,
)
from transformers.testing_utils import CaptureLogger
import transformers
import logging
import warnings
import sys
import os
from arguments import ModelArguments, DatasetArguments, GenerationArguments
from stopping_criterias import BraceMatchingCriteria
import re
import time


from accelerate.logging import get_logger
get_logger("transformers").setLevel(logging.INFO)
logger = get_logger(__name__)


def main():
    """
    Generate new data by sampling from the original data.
    """
    # See all possible arguments in arguments/*.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DatasetArguments, GenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, gen_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Initialize accelerator
    accelerator = Accelerator()

    if gen_args.seed is not None:
        set_seed(gen_args.seed)

    
    # output file
    i = "{:05n}".format(accelerator.process_index + 1)
    n = "{:05n}".format(accelerator.num_processes)
    path = Path(gen_args.output_dir) / (f"{i}-of-{n}" + f".{data_args.dataset_split}.jsonl")


    # Write the generation config to disk
    if accelerator.is_main_process:
        if os.path.isdir(gen_args.output_dir):
            if gen_args.overwrite_output_dir:
                if len(os.listdir(gen_args.output_dir)) > 0:
                    logger.warning(f"Output directory ({gen_args.output_dir}) already exists and is not empty. Overwriting it.")
                    # Add countdown for safety
                    for i in range(5, 0, -1):
                        logger.warning(f"***** Overwriting output directory in {i} seconds *****")
                        time.sleep(1)

                    os.system(f"rm -r {gen_args.output_dir}/*")
        
        if gen_args.output_dir is not None:    
            #safe_dataset_name = urllib.parse.quote(args.dataset_name, safe='')
            #urlencode args.dataset_name
            #safe_model_name = urllib.parse.quote(args.model_name_or_path, safe='')
            Path(gen_args.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Need a output directory.")
    accelerator.wait_for_everyone()


    skip_indices = []
    if not gen_args.overwrite_output_dir:
        if gen_args.id_column_name is None:
            raise ValueError("id_column_name must be set to use resumable generation.")
        
        with accelerator.main_process_first():
            # try to load existing files 
            # check if the file is already generated
            regex = r"^(\d+)-of-(\d+)." + re.escape(data_args.dataset_split) + r"\.jsonl$"
            files = os.listdir(gen_args.output_dir)

            skip_indices = set()
            for file in files:
                matches = re.search(regex, file)
                if matches:
                    with open(Path(gen_args.output_dir) / file, "r") as f:
                        for line in f:
                            entry = json.loads(line)
                            skip_indices.add(entry["id"])
        accelerator.wait_for_everyone()

    # Load the dataset
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.dataset_split)#, revison=data_args.dataset_revision)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, revision=model_args.model_revision)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, padding_side='left')
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, padding_side='left')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.distributed_type == DistributedType.NO \
       or (accelerator.distributed_type == DistributedType.MULTI_GPU and accelerator.num_processes <= 1):
        device_map = "auto" # Activate the naive model parallelism.
    else:
        device_map = None

    torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.tie_weights()

    # Write the model config and generation config to disk
    if accelerator.is_main_process:
        # Dump the model config without defaults to disk
        with open( Path(gen_args.output_dir) / "model_config_diff.json", "w") as f:
            json.dump(config.to_diff_dict(), f, indent=4)

        # Dump the model config with defaults to disk
        with open(Path(gen_args.output_dir) / "model_config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=4)
    
    if model_args.adapter_name_or_path is not None:
        from peft import PeftModel, PeftConfig
        adapter_config = PeftConfig.from_pretrained(model_args.adapter_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.adapter_name_or_path)
        
        #only if lora is used:
        if adapter_config.peft_type in ("LORA", "IA3"):
            print("merging adapter...")
            model = model.merge_and_unload() # merge the adapter into the model for faster inference.
        
        # Write the adapter config and generation config to disk
        if accelerator.is_main_process:
            print(adapter_config)
            # Dump the adapter configto disk
            with open(Path(gen_args.output_dir) / "adapter_config.json", "w") as f:
                json.dump(adapter_config.to_dict(), f, indent=4, default=lambda x: None)


    generation_config = {}
    if gen_args.generation_config_file is not None:
        # read from file
        with open(gen_args.generation_config_file, "r") as f:
            generation_config = json.load(f)
            generation_config = GenerationConfig.from_dict(generation_config)

    elif model_args.model_name_or_path:
        generation_config = model.generation_config
        logger.warning(f"Using default generation config from model: {generation_config}")

    if gen_args.max_new_tokens is not None:

        if type(gen_args.max_new_tokens) == int:
            generation_config.max_new_tokens = gen_args.max_new_tokens
        elif gen_args.max_new_tokens == "auto":
            generation_config.max_new_tokens = None
    
        logger.info(f"max_new_tokens are set to {gen_args.max_new_tokens}")
    else:
        raise ValueError("max_new_tokens is not set.")

    # Write the generation config to disk
    if accelerator.is_main_process:
        print(generation_config)

        # Dump the generation config without defaults to disk
        with open(Path(gen_args.output_dir) / "generation_config_diff.json", "w") as f:
            json.dump(generation_config.to_diff_dict(), f, indent=4)

        # Dump the generation config with defaults to disk
        with open(Path(gen_args.output_dir) / "generation_config.json", "w") as f:
            json.dump(generation_config.to_dict(), f, indent=4)




    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if gen_args.block_size is None:
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
        if gen_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({gen_args.block_size}) seems to be larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={gen_args.block_size}."
            )
        # Some models have inproperly tokenizer.model_max_length, so we allow overriding it
        #block_size = min(data_args.block_size, tokenizer.model_max_length)
        block_size = gen_args.block_size

    if model_args.adapter_name_or_path and adapter_config.peft_type == "PROMPT_TUNING":
        block_size -= adapter_config.num_virtual_tokens


    if gen_args.max_window_size is None:
        gen_args.max_window_size = block_size - int(generation_config.max_new_tokens or 0) #TODO: check if this should be 1


    # Define stopping criterias
    stopping_criteria_list = StoppingCriteriaList()
    if gen_args.use_brace_matching:
        # Expensive to initialize, so reuse the same instance.
        stopping_criteria_list.append(BraceMatchingCriteria(tokenizer, gen_args.brace_matching_start_level))
    


    # Preprocessing the datasets.
    keep_columns = []
    if gen_args.keep_columns is not None:
        keep_columns = gen_args.keep_columns

    text_column_name = data_args.text_column_name
    reference_column_name = data_args.reference_column_name
    keep_columns.append(reference_column_name)
    min_input_length = 0
    max_src_length = block_size - 1 # Minimum room for one token

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
    # Tokenize the data
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            input_ids = tokenizer(examples[text_column_name])["input_ids"]
            attention_mask = tokenizer(examples[text_column_name])["attention_mask"]
            reference_input_ids = tokenizer(examples[reference_column_name])["input_ids"]
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "reference_input_ids": reference_input_ids}


    def filter_length_function(examples):
        res = []
        is_dropped = 0
        for input_ids, reference_input_ids in zip(examples["input_ids"], examples["reference_input_ids"]):
            if len(input_ids) < min_input_length:
                res.append(False)
                is_dropped += 1
            elif (len(input_ids) + len(reference_input_ids)) > max_src_length:
                res.append(False)
                is_dropped += 1
            else:
                res.append(True)
        if is_dropped:
            logger.warning(
                f"Dropped {is_dropped} examples because they were shorter than {min_input_length} tokens, or larger than (or equal to) the max position embeddings allowed "
                f"({block_size})."
            )
        return res
    
    def filter_subsamples_function(examples):
        # resumable_dataset. If existing generation is found, skip the example.
        res = []
        for id in examples[gen_args.id_column_name]:
            if id in skip_indices:
                res.append(False)
            else:
                res.append(True)
        return res


    def single_batch_function(examples, indices):
        new_examples = {
            "id": [],
            "part": [],
            "input_ids": [],
            "attention_mask": [],
            "reference_input_ids": [],
        }

        for i, id in enumerate(indices):
            if gen_args.id_column_name is not None:
                id = examples[gen_args.id_column_name][i]

            input_ids = examples["input_ids"][i][-gen_args.max_window_size:]
            mask = examples["attention_mask"][i][-gen_args.max_window_size:]
            reference_input_ids = examples["reference_input_ids"][i]

            new_examples["id"].append(id)
            new_examples["part"].append([1,1])
            new_examples["input_ids"].append(input_ids)
            new_examples["attention_mask"].append(mask)
            new_examples["reference_input_ids"].append(reference_input_ids)
        return new_examples


    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        filtered_dataset = tokenized_dataset.filter(
            filter_length_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Filtering min length",
        )
        resumable_dataset = filtered_dataset.filter(
            filter_subsamples_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Filtering subsamples",
        )
        minibatch_dataset = resumable_dataset.map(
            single_batch_function,
            with_indices=True,
            batched=True,
            remove_columns=[c for c in filtered_dataset.column_names if c not in keep_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preparing minibatches",
        )
    
    dataset = minibatch_dataset
    
    def data_collator(examples):
        batch = tokenizer.pad(examples).data
        # Access data dict because we keep columns that are not to be put on device.
        # BatchEncoding has "to" func and need to circumvent  https://github.com/huggingface/accelerate/pull/2438
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        return batch

    # Create the DataLoader
    data_loader = DataLoader(dataset, shuffle=False,
                             collate_fn=data_collator, batch_size=gen_args.per_device_batch_size)
    model.eval()
    
    # Optimization libraries
    if gen_args.use_deepspeed_inference:
        import deepspeed
        # init deepspeed inference engine
        model = deepspeed.init_inference(
            model=model,      # Transformers models
            tensor_parallel={"tp_size": accelerator.num_processes},
            dtype=torch_dtype, # dtype of the weights
            replace_with_kernel_inject=False, # replace the model with the kernel injector
            #quantize_bits=16, # quantize the model to 32 bits
        )
        model = model.module
        
    else:
        # Prepare everything with `accelerator`.
        model, data_loader = accelerator.prepare(model, data_loader)

    fp = open(path, 'a')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(total=(len(data_loader) * gen_args.per_device_batch_size) + len(skip_indices), initial=len(skip_indices), position=accelerator.process_index) #,disable=not accelerator.is_local_main_process)
    for batch in data_loader:
        prompt_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        
        if generation_config.max_new_tokens is None:
            max_new_tokens = block_size - prompt_ids.shape[-1]
        else:
            max_new_tokens = generation_config.max_new_tokens
        #accelerator.print("Generating...")
        #generation_config.num_return_sequences = 2
        with torch.no_grad():
            # generate the data
            generated = accelerator.unwrap_model(model).generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria_list,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                #synced_gpus=True, #not needed as of https://github.com/huggingface/transformers/pull/22242
                do_sample=False, #TODO: turn off !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            )
        # decode the data

        decoded_prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        
        #predicted_ids = generated[:, -max_new_tokens:]
        predicted_ids = generated[:, prompt_ids.shape[-1]:]
        decoded_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        decoded_reference = batch[reference_column_name]

        progress_bar.update(gen_args.per_device_batch_size)

        # save the data to disk
        for index in range(generated.shape[0]):
            b_index = index//generation_config.num_return_sequences # original batch index

            entry = {}
            entry["id"] = batch["id"][b_index]
            entry["part"] = batch["part"][b_index]
            entry["seq"] = [(index % generation_config.num_return_sequences) + 1, generation_config.num_return_sequences]
            entry["prompt"] = decoded_prompts[b_index]
            entry["reference"] = decoded_reference[b_index]
            entry["prediction"] = decoded_predictions[index]

            ended = None
            
            if len(predicted_ids[index]) == max_new_tokens:
                ended = "length"
            for stopping_criteria in stopping_criteria_list: # possible race condition
                if stopping_criteria.is_sequence_stopped(index):
                    ended = stopping_criteria.stop_reason()
            if predicted_ids[index][-1].item() == tokenizer.eos_token_id:
                ended = "stop"
            entry["finish_reason"] = ended

            entry["meta"] = { "subset": data_args.dataset_config_name }

            # keep all "keep_columns":
            if gen_args.keep_columns is not None:
                for colname in gen_args.keep_columns:
                    entry[colname] = batch[colname][b_index]

            fp.write(json.dumps(entry) + "\n")
            fp.flush()
        
        if gen_args.use_brace_matching:
            stopping_criteria_list[0].reset() # reset the BraceMatchingCriteria


    fp.close()

    accelerator.wait_for_everyone()
    
    progress_bar.close()


if __name__ == "__main__":
    main()