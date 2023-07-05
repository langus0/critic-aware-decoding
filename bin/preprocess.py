#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

from data import get_dataset_class_by_name

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Load the raw dataset using a loader specified in `data.py`,
    process it, and save it in the `./data/{output_dir}` directory.

    By default, a directory with processed dataset will contain the files `train.json`, `dev.json`, `test.json`,
    each file with the following structure:
    {
        "data" : [
            {... data entry #1 ...},
            {... data entry #2 ...},
            .
            .
            .
            {... data entry #N ...},
        ]
    }
    This format is expected for loading the data into PyTorch dataloaders for training and inference.
    """

    def __init__(self, dataset, out_dirname, mode):
        self.dataset = dataset
        self.out_dirname = out_dirname
        self.mode = mode

    def linearize_triples(self, triples):
        """
        A simple way of linearizing triples which does not require adding special tokens to the model vocabulary.
        Although it helps to avoid incompatible sizes in model vocabulary, this should be improved in the future.
        """
        X_SEP = " | "
        T_SEP = " ▸ "
        out = []

        for t in triples:
            out.append(t.subj + X_SEP + t.pred + X_SEP + t.obj)

        out = T_SEP.join(out)

        return out

    def create_examples(self, entry, dataset):
        """
        Generates training examples from an entry in the dataset
        """
        examples = []

        for ref in entry.refs:
            if self.mode == "plain":
                # creating a single example without any extra processing
                example = {"in": entry.data, "out": ref}
                examples.append(example)
            elif self.mode == "linearize_triples":
                example = {"in": self.linearize_triples(entry.data), "out": ref}
                examples.append(example)
            elif self.mode == "linearize_triples_align":
                example = {
                    "in": self.linearize_triples(entry.data),
                    "out": ref,
                    # i - input triples r - sentence number in reference
                    # alignments are number from 1 onwards
                    # 0-X is reserved for alignments with no input triple
                    # X-0 is reserved for triples without no realization in reference
                    "align": ", ".join([f"{i}-{r}" for i, r in entry.align]),
                    # number of sentences in reference prompt
                    "num_ref_sentences": entry.num_ref_sentences,
                }
                examples.append(example)
            else:
                raise ValueError("Unknown mode")

        return examples

    def process(self, split):
        output = {"data": []}
        data = self.dataset.data[split]

        for i, entry in enumerate(data):
            examples = self.create_examples(entry, dataset)

            if examples and split != "train":
                # keep just one example in dev/test sets
                examples = [examples[0]]

            for example in examples:
                output["data"].append(example)

        with open(os.path.join(self.out_dirname, f"{split}.json"), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to be loaded, refers to the class attribute `name` of the class in `data.py`",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory from which the dataset should be loaded.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory in which the processed dataset should be saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "linearize_triples", "linearize_triples_align"],
        required=True,
        help="Preprocessing mode",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Dataset splits (e.g. train dev test)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    random.seed(args.seed)
    logger.info(args)

    # locate the class in `data.py` with the class attribute `name` corresponding to the argument
    dataset = get_dataset_class_by_name(args.dataset)()

    try:
        dataset.load(splits=args.splits, path=args.input_dir)
    except FileNotFoundError as err:
        logger.error("Dataset could not be loaded")
        raise err

    try:
        out_dirname = args.output_dir
        os.makedirs(out_dirname, exist_ok=True)
    except OSError as err:
        logger.error(f"Output directory {out_dirname} can not be created")
        raise err

    preprocessor = Preprocessor(dataset=dataset, out_dirname=out_dirname, mode=args.mode)
    for split in args.splits:
        preprocessor.process(split)

    logger.info("Preprocessing finished.")
