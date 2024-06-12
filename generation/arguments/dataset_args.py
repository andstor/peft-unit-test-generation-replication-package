from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default=None, metadata={"help": "The dataset split to use."}
    )
    text_column_name: str = field(
        default=None, metadata={"help": "The dataset column name to use."}
    )
    reference_column_name: str = field(
        default=None, metadata={"help": "The dataset column name to use as reference for the target sequence."}
    )
    dataset_revision: str = field(
        default="main",
        metadata={"help": "The specific dataset version to use (can be a branch name, tag name or commit id)."},
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there is no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    
    
    #keep_linebreaks: bool = field(
    #    default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    #)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")

