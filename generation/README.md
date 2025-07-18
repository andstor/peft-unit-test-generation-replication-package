# Generation
## Description
This directory contains script for generating text with language models. The scripts are designed to be used with the Hugging Face `transformers` library and the `datasets` library. Accelerate is used to speed up the generation process.

## Requirements

### Dependencies
Install the Python dependencies defined in the requirements.txt.
```bash
pip install -r requirements.txt
```

### Accelerate
Setup accelerate:
```bash
accelerate config
```

## Generation with Hugging Face models
The `run_gen.py` script will generate samples from a specified dataset with a Hugging Face model. It supports a large set of options for controlling the generation process.

### Usage

```bash
usage: run_gen.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--model_type MODEL_TYPE] [--config_name CONFIG_NAME]
                  [--tokenizer_name TOKENIZER_NAME] [--use_fast_tokenizer [USE_FAST_TOKENIZER]] [--no_use_fast_tokenizer]
                  [--model_revision MODEL_REVISION] [--token TOKEN] [--use_auth_token [USE_AUTH_TOKEN]]
                  [--adapter_name_or_path ADAPTER_NAME_OR_PATH] [--torch_dtype {auto,bfloat16,float16,float32}]
                  [--dataset_name DATASET_NAME] [--dataset_config_name DATASET_CONFIG_NAME] [--dataset_split DATASET_SPLIT]
                  [--text_column_name TEXT_COLUMN_NAME] [--reference_column_name REFERENCE_COLUMN_NAME]
                  [--dataset_revision DATASET_REVISION] [--streaming [STREAMING]] [--overwrite_cache [OVERWRITE_CACHE]]
                  [--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE]
                  [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS] [--generation_config_file GENERATION_CONFIG_FILE]
                  [--per_device_batch_size PER_DEVICE_BATCH_SIZE] [--output_dir OUTPUT_DIR]
                  [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]] [--id_column_name ID_COLUMN_NAME]
                  [--keep_columns KEEP_COLUMNS [KEEP_COLUMNS ...]] [--seed SEED] [--max_new_tokens MAX_NEW_TOKENS]
                  [--max_window_size MAX_WINDOW_SIZE] [--block_size BLOCK_SIZE] [--use_brace_matching [USE_BRACE_MATCHING]]
                  [--brace_matching_start_level BRACE_MATCHING_START_LEVEL]
                  [--use_deepspeed_inference [USE_DEEPSPEED_INFERENCE]]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Do not set if you want to train a model from scratch.
                        (default: None)
  --model_type MODEL_TYPE
                        If training from scratch, pass a model type from the list: bart, bert, bert-generation, big_bird,
                        bigbird_pegasus, biogpt, blenderbot, blenderbot-small, bloom, camembert, llama, codegen, cohere,
                        cpmant, ctrl, data2vec-text, dbrx, electra, ernie, falcon, fuyu, gemma, git, gpt2, gpt2, gpt_bigcode,
                        gpt_neo, gpt_neox, gpt_neox_japanese, gptj, jamba, jetmoe, llama, mamba, marian, mbart, mega, megatron-
                        bert, mistral, mixtral, mpt, musicgen, musicgen_melody, mvp, olmo, open-llama, openai-gpt, opt,
                        pegasus, persimmon, phi, phi3, plbart, prophetnet, qdqbert, qwen2, qwen2_moe, recurrent_gemma,
                        reformer, rembert, roberta, roberta-prelayernorm, roc_bert, roformer, rwkv, speech_to_text_2, stablelm,
                        starcoder2, transfo-xl, trocr, whisper, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet,
                        xmod (default: None)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name (default: None)
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: True)
  --no_use_fast_tokenizer
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: False)
  --model_revision MODEL_REVISION
                        The specific model version to use (can be a branch name, tag name or commit id). (default: main)
  --token TOKEN         The token to use as HTTP bearer authorization for remote files. If not specified, will use the token
                        generated when running `huggingface-cli login` (stored in `~/.huggingface`). (default: None)
  --use_auth_token [USE_AUTH_TOKEN]
                        The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.
                        (default: None)
  --adapter_name_or_path ADAPTER_NAME_OR_PATH
                        The name or path of the adapter to use. (default: None)
  --torch_dtype {auto,bfloat16,float16,float32}
                        Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype
                        will be automatically derived from the model's weights. (default: None)
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library). (default: None)
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the datasets library). (default: None)
  --dataset_split DATASET_SPLIT
                        The dataset split to use. (default: None)
  --text_column_name TEXT_COLUMN_NAME
                        The dataset column name to use. (default: None)
  --reference_column_name REFERENCE_COLUMN_NAME
                        The dataset column name to use as reference for the target sequence. (default: None)
  --dataset_revision DATASET_REVISION
                        The specific dataset version to use (can be a branch name, tag name or commit id). (default: main)
  --streaming [STREAMING]
                        Enable streaming mode (default: False)
  --overwrite_cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets (default: False)
  --validation_split_percentage VALIDATION_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set in case there is no validation split (default: 5)
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing. (default: None)
  --generation_config_file GENERATION_CONFIG_FILE
                        Generation config path if not the same as model_name. (default: None)
  --per_device_batch_size PER_DEVICE_BATCH_SIZE
                        Batch size (per device) for generation. (default: 8)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use this to continue training if output_dir points to a
                        checkpoint directory. (default: False)
  --id_column_name ID_COLUMN_NAME
                        The column name of the dataset to use as id. If not provided, the index will be used. (default: None)
  --keep_columns KEEP_COLUMNS [KEEP_COLUMNS ...]
                        The column names of the dataset to keep separate by commas. If not provided, all columns will be removed.
                        (default: None)
  --seed SEED           Seed for random number generation. (default: None)
  --max_new_tokens MAX_NEW_TOKENS
                        The maximum number of new tokens to generate. (default: None)
  --max_window_size MAX_WINDOW_SIZE
                        The maximum number of tokens in the input. (default: None)
  --block_size BLOCK_SIZE
                        Optional limit the model's max position embeddings. (default: None)
  --use_brace_matching [USE_BRACE_MATCHING]
                        Whether to use brace matching as a stopping criteria. (default: False)
  --brace_matching_start_level BRACE_MATCHING_START_LEVEL
                        The level of brace matching to start from. (default: 0)
  --use_deepspeed_inference [USE_DEEPSPEED_INFERENCE]
                        Whether to use deepspeed as an optimization library. (default: False)
```

### Complete generation

The script supports sequence to sequence objective within the causal language modeling paradigm. To use this, simply provide both a `--text_column_name` and a `--reference_column_name` argument. The `--text_column_name` argument should be the names of the column that contain the input text. The `--reference_column_name` argument should be the name of the column that contains the target text. 

Any inputs, meaning `text_column_name` + `reference_column_name`, that are longer than the `--block_size` will be dropped. The `--block_size` argument is used to limit the model's maximum position embeddings.
By setting `--max_new_tokens` to `auto`, all the unused embedding space is used to generate as much as possible up to the `--block_size`. The maximum number of tokens in the input can be truncated (from the left) by setting `--max_window_size`, thus allowing for a longer output (`--max_new_tokens`). 


#### Example
The following example will generate samples from the test split of the [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x) dataset using the greedy decoding strategy. The output will be saved to the `output` directory.

```bash
accelerate launch run_gen.py \
--model_name_or_path gpt2 \
--torch_dtype auto \
--dataset_name THUDM/humaneval-x \
--dataset_config_name java \
--dataset_split test \
--text_column_name prompt \
--reference_column_name canonical_solution \
--id_column_name task_id \
--block_size 768 \
--per_device_batch_size 4 \
--output_dir .output \
--seed 42 \
--max_new_tokens "auto"
```

#### Early stopping
Early stopping can be done by using brace matching. This will stop the generation when the number of open braces is equal to the number of closed braces. The level of brace matching to start from can be controlled by the `--brace_matching_start_level` argument. The following example will use brace matching as a stopping criteria and start from level 1, meaning one open brace is already present in the prompt.

```bash
--use_brace_matching \
--brace_matching_start_level 1
```

### Parameter-Efficient Fine-Tuning (PEFT) methods

Currently supported PEFT methods are LoRA, (IA)^$3$, and Prompt Tuning. To load an adapter module, use the `--adapter_name_or_path` argument and pass the path to the adapter modules directory.

```bash
--adapter_name_or_path path/to/adapter \
```
