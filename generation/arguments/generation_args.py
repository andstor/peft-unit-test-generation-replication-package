from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class GenerationArguments:
    """
    Arguments pertaining to generation.
    """
    generation_config_file: Optional[str] = field(
        default=None, metadata={"help": "Generation config path if not the same as model_name."}
    )
    per_device_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size (per device) for generation."}
    )
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    id_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the dataset to use as id. If not provided, the index will be used."}
    )
    keep_columns: Optional[List[str]] = field(
        default=None, metadata={"help": "The column names of the dataset to keep separate by commas. If not provided, all columns will be removed."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "Seed for random number generation."}
    )
    max_new_tokens: Optional[str] = field(
        default=None, metadata={"help": "The maximum number of new tokens to generate."}
    )
    max_window_size: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of tokens in the input."}
    )
    block_size: Optional[int] = field(
        default=None, metadata={"help": ("Optional limit the model's max position embeddings.")}
    )

    # Stopping criterias
    use_brace_matching: bool = field(
        default=False, metadata={"help": "Whether to use brace matching as a stopping criteria."}
    )
    brace_matching_start_level: Optional[int] = field(
        default=0, metadata={"help": "The level of brace matching to start from."}
    )

    # Optimization libraries

    use_deepspeed_inference: bool = field(
        default=False, metadata={"help": "Whether to use deepspeed as an optimization library."}
    )


    def __post_init__(self):
        if self.max_new_tokens is not None and self.max_new_tokens.isdigit():
            self.max_new_tokens = int(self.max_new_tokens)
        
        if self.keep_columns is not None:
            self.keep_columns = self.keep_columns.split(",")

        #if self.use_brace_matching and self.per_device_batch_size > 1:
        #    raise ValueError("Brace matching can only be used with a per_device_batch_size of 1.")
