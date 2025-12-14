from typing import Any

import pandas as pd
from transformers.integrations import WandbCallback


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows visualization of the
    model predictions as training progresses.

    Attributes:
        trainer: Hugging Face Trainer instance.
        tokenizer: Tokenizer associated with the model.
        sample_dataset: Subset of the validation dataset for predictions.
        num_samples: Number of samples to select from validation for predictions.
        freq: Frequency of logging (epochs).
    """

    def __init__(
        self,
        trainer: Any,
        tokenizer: Any,
        val_dataset: Any,
        num_samples: int = 100,
        freq: int = 2,
    ):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer: Hugging Face Trainer instance.
            tokenizer: Tokenizer associated with the model.
            val_dataset: Validation dataset.
            num_samples: Number of samples to select from validation
                for generating predictions. Defaults to 100.
            freq: Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.take(num_samples)
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        epoch = getattr(state, "epoch", -1)
        if epoch % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})
