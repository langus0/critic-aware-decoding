#!/usr/bin/env python3

import argparse
import logging
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from dataloader import Seq2SeqDataModule
from inference import CriticGenDataInferenceModule, Seq2SeqInferenceModule, CriticAwareInferenceModule
from wandb_logging import save_predictions

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Seq2SeqDataModule.add_argparse_args(parser)

    parser.add_argument(
        "--exp_dir",
        default="experiments",
        type=str,
        help="Base directory of the experiment.",
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name.")
    parser.add_argument("--model_name", type=str, help="Start with vanilla model instead of a trained model.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size used for decoding.")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory with the data.")
    parser.add_argument("--split", type=str, required=True, help="Split to decode (dev / test).")
    parser.add_argument("--wrapper", type=str, required=False,
                        help="Logit wrapper used during decoding (classifier/data)")
    parser.add_argument("--condition_lambda", type=float, required=False,
                        help="condition_lambda used in critic-driven decoding")
    parser.add_argument("--critic_top_k", type=int, required=False,
                        help="numer of words to analyze used in critic-driven decoding")
    parser.add_argument("--critic_ckpt", type=str, required=False, help="checkpoint with critic classifier")
    parser.add_argument("--linear_warmup", action="store_true", help="turns on linear warmup during decoding")

    parser.add_argument(
        "--out_filename",
        type=str,
        default=None,
        help="Override the default output filename <split>.out.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.",
    )
    parser.add_argument("--max_threads", default=8, type=int, help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size used for decoding.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum number of tokens per example",
    )
    # parser.add_argument("--verbose", action="store_true", help="Show outputs during generation.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Use 8-bit precision. Packages `bitsandbytes` and `accelerate` need to be installed.")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    out_path = os.path.join(args.exp_dir, args.experiment, f"{args.split}.out")

    if args.model_name:
        os.makedirs(os.path.join(args.exp_dir, args.experiment), exist_ok=True)
        logger.info(f"Loading vanilla {args.model_name}")
        if args.wrapper:
            di = CriticGenDataInferenceModule(args, None, None, wrapper=args.wrapper)
        else:
            di = Seq2SeqInferenceModule(args)

    elif args.experiment:
        model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
        if args.wrapper:
            if args.wrapper == "data":
                di = CriticGenDataInferenceModule(args, None, None, wrapper=args.wrapper)
            elif args.wrapper == "data-full":
                args.wrapper = "data"
                di = CriticAwareInferenceModule(args, None, None, wrapper=args.wrapper)
            else:
                critic_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                from critics.lmodule import ClassificationModule
                from critics.model import SimpleClassifierModelWithBNSELU

                critic_model = ClassificationModule.load_from_checkpoint(checkpoint_path=args.critic_ckpt,
                                                                         model=SimpleClassifierModelWithBNSELU(
                                                                             'xlm-roberta-base'))
                critic_model.eval()
                di = CriticAwareInferenceModule(args, critic_model, critic_tokenizer, model_path=model_path,
                                                wrapper=args.wrapper)
        else:
            di = Seq2SeqInferenceModule(args, model_path=model_path)

    dm = Seq2SeqDataModule(args, model_name=di.model_name)

    dm.prepare_data()
    dm.setup("predict")
    trainer = pl.Trainer.from_argparse_args(args)

    # # TODO cleaner way
    # di.model.verbose = args.verbose

    di.model.tokenizer = dm.tokenizer
    di.model.beam_size_decode = args.beam_size

    dataloader_map = {"train": dm.train_dataloader, "dev": dm.val_dataloader, "test": dm.test_dataloader}
    # run the batch decoding
    predictions = trainer.predict(dataloaders=[dataloader_map[args.split]()], model=di, return_predictions=True)

    out_filename = args.out_filename or f"{args.split}.out"

    if args.wrapper == "data":
        import csv

        with open(os.path.join(args.exp_dir, args.experiment, out_filename + "-data"), "w", encoding="UTF-8") as fw:
            writer = csv.writer(fw)
            for batch in predictions:
                for p, h in batch["data"]:
                    writer.writerow([str(p), str(h), 0])

    with open(out_filename + '.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.exp_dir, args.experiment, out_filename), "w") as f:
        save_predictions(f, predictions)
