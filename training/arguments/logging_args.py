from dataclasses import dataclass, field

@dataclass
class LoggingArguments:
    """
    Arguments pertaining to logging.
    """
    log_preditions: bool = field(
        default=False, metadata={"help": "Whether to log predictions during training."}
    )
    log_predition_samples: int = field(
        default=10, metadata={"help": "Number of samples to log during training."}
    )
