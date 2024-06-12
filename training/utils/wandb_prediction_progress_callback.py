from transformers.integrations import WandbCallback
import pandas as pd
import os
import numpy as np

from utils import decode_predictions

#os.environ["WANDB_LOG_MODEL"] = "checkpoint"

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Control the frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq


    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        #if state.epoch % self.freq == 0:
                # generate predictions
                #print("OMG")
                #print(self.sample_dataset)
        predictions = self.trainer.predict(self.sample_dataset)
        if state.global_step % state.eval_steps == 0:
            if state.is_world_process_zero:
                # decode predictions and labels
                #print(predictions)
                predictions = decode_predictions(self.tokenizer, predictions)
                # add predictions to a wandb.Table
                predictions_df = pd.DataFrame(predictions)
                predictions_df["global_step"] = state.global_step
                records_table = self._wandb.Table(dataframe=predictions_df)
                # log the table to wandb
                self._wandb.log({"sample_predictions": records_table})
