#!/usr/bin/env python3

import argparse
import logging

import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

logger = logging.getLogger(__name__)


class TrainingModule(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["datamodule"])
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        self.datamodule = kwargs.get("datamodule", None)

    def forward(self, **inputs):
        out = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"]
        )
        return {"loss": out["loss"], "logits": out["logits"]}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log("loss/val", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

        max_steps = -1 if self.args.max_steps is None else self.args.max_steps
        max_epochs = int(self.args.max_epochs)

        total_steps = max_epochs * len(self.datamodule.train_dataloader()) if max_steps == -1 else max_steps
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "polynomial", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logger.info("Using Adam optimizer with polynomial scheduling")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser


class Seq2SeqTrainingModule(TrainingModule):
    MODEL_OUT = "model_out"
    TOKENIZER_OUT = "tokenizer_out"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        if hasattr(args, "load_in_8bit") and args.load_in_8bit:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto", load_in_8bit=True,
                                                               return_dict=True)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, return_dict=True)

        mem = self.model.get_memory_footprint()
        logger.info(f"Memory consumed by the model: {mem} bytes")

    def _predict_step(self, batch, batch_idx):
        model_out = self.model.generate(
            batch["input_ids"], max_length=self.args.max_length, num_beams=1, num_return_sequences=1
        )

        tokenizer_out = self.tokenizer.batch_decode(
            model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.info(tokenizer_out[0])

        return {self.MODEL_OUT: model_out, self.TOKENIZER_OUT: tokenizer_out}

    def test_step(self, batch, batch_idx):
        # TODO run evaluation because test step should have available ground truth!
        return self._predict_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self._predict_step(batch, batch_idx)
