#!/usr/bin/env python3

import logging
from collections import namedtuple

from datasets import load_dataset

from text import normalize

logger = logging.getLogger(__name__)

RDFTriple = namedtuple("RDFTriple", ["subj", "pred", "obj"])


def get_dataset_class_by_name(name):
    """
    A helper function which allows to use the class attribute `name` of a Dataset
    (sub)class as a command-line parameter for loading the dataset.
    """
    try:
        # case-insensitive
        available_classes = {
            o.name.lower(): o for o in globals().values() if type(o) == type(Dataset) and hasattr(o, "name")
        }
        return available_classes[name.lower()]
    except AttributeError:
        logger.error(
            f"Unknown dataset: '{name}'. Please create \
            a class with an attribute name='{name}' in 'data.py'. \
            Available classes: {available_classes}"
        )
        return None


class DataEntry:
    """
    An entry in the dataset

    Alignment align: is used for triple data type.
        Example alignment "1-1,2-1, 3-2, 4-2" describes that
        first two triples are aligned to first sentence in a reference
        and that triples three and four are aligned to second sentence in the reference.
        Note that sentence in reference as well as triples are numbered from one onward
        because zero is reserved for marking no alignment.

        num_ref_sentences is used for triple data type when alignment is present.
            It marks the number of sentences in the reference.
            It could be used for sanity checks for sentence segmentations.
    """

    def __init__(self, data, refs, data_type, align=None, num_ref_sentences=None):
        self.data = data
        self.refs = refs
        self.data_type = data_type
        self.align = align
        self.num_ref_sentences = num_ref_sentences

    def __repr__(self):
        return str(self.__dict__)


class Dataset:
    """
    Base class for the datasets
    """

    def __init__(self):
        self.data = {split: [] for split in ["train", "dev", "test"]}

    def load(self, splits, path=None):
        """
        Load the dataset. Path can be specified for loading from a directory
        or omitted if the dataset is loaded from HF.
        """
        raise NotImplementedError


class WebNLG(Dataset):
    """
    The WebNLG dataset: https://gem-benchmark.com/data_cards/web_nlg
    Contains RDF triples from DBPedia and their crowdsourced verbalizations.
    """

    name = "webnlg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, splits, path):
        # load the dataset from HF datasets
        dataset = load_dataset("gem", "web_nlg_en")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                if split == "test":
                    refs = example["references"]
                else:
                    refs = [example["target"]]

                entry = DataEntry(data=triples, refs=refs, data_type="triples")
                self.data[split].append(entry)
