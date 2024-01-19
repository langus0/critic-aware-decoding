import logging
from pathlib import Path
from typing import Dict, List, Any

from pytorch_lightning.loggers.wandb import WandbLogger

from decoding.model import Seq2SeqTrainingModule

TOKENIZER_OUT = Seq2SeqTrainingModule.TOKENIZER_OUT


def upload_predictions(logger: WandbLogger, predictions: List[Dict[str, Any]], name: str):
    ckpt_dir = Path(logger.save_dir)
    prediction_file = ckpt_dir / f"{name}_predictions.txt"

    with open(prediction_file, "wt") as f:
        save_predictions(f, predictions)
    logger.experiment.save(str(prediction_file))


def save_predictions(file_handle, predictions: List[Dict[str, Any]]):
    """See nlg/model:Seq2SeqTrainingModule._predict_step"""
    n = 0
    for batch in predictions:
        b_outs = batch[TOKENIZER_OUT]
        for o in b_outs:
            logging.info(f"[{n}] {o}")
            n += 1
            file_handle.write(f"{o}\n")
