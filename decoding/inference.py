#!/usr/bin/env python3

import itertools
import logging

import pytorch_lightning as pl
import torch
from transformers import (
    AutoTokenizer,
)
from transformers.generation import LogitsProcessor, LogitsProcessorList

from model import Seq2SeqTrainingModule

logger = logging.getLogger(__name__)


class InferenceModule(pl.LightningModule):
    def __init__(self, args, training_module_cls, model_path=None):
        super().__init__()
        self.args = args
        self.beam_size = args.beam_size if hasattr(args, "beam_size") else None

        if model_path is not None:
            self.model = training_module_cls.load_from_checkpoint(model_path)
            self.model.freeze()
            self.model_name = self.model.model.name_or_path
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model_name = args.model_name
            self.model = training_module_cls(args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def predict(self, s, beam_size=None):
        inputs = self.tokenizer(s, return_tensors="pt", truncation=True)

        # if hasattr(self.args, "gpus") and self.args.gpus > 0:
        #     self.model.cuda()
        #     for key in inputs.keys():
        #         inputs[key] = inputs[key].cuda()
        # else:
        #     logger.warning("Not using GPU")

        return self.generate(inputs["input_ids"], beam_size)

    def forward(self, **inputs):
        return self.model(inputs)

    def generate(self, input_ids):
        raise NotImplementedError


class Seq2SeqInferenceModule(InferenceModule):
    def __init__(self, args, model_path=None):
        super().__init__(args, model_path=model_path, training_module_cls=Seq2SeqTrainingModule)

    def generate(self, input_ids, beam_size=None):
        if not beam_size:
            beam_size = self.beam_size

        input_ids = input_ids.to(self.model.model.device)
        out = self.model.model.generate(
            input_ids,
            max_length=self.args.max_length,
            num_beams=beam_size,
            num_return_sequences=beam_size,
        )
        sentences = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return sentences

    def predict_step(self, batch, batch_idx):
        model_out = self.model.model.generate(
            batch["input_ids"],
            max_length=self.args.max_length,
            num_beams=self.beam_size,
            num_return_sequences=1,
        )

        tokenizer_out = self.tokenizer.batch_decode(
            model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.info(tokenizer_out[0])

        return {self.model.MODEL_OUT: model_out, self.model.TOKENIZER_OUT: tokenizer_out}


class TopKLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class GeneralCriticLogitsWarper(LogitsProcessor):
    def __init__(self, model, conditioning_model, conditioning_tokenizer, inputs, top_k: int,
                 filter_value: float = -float("Inf"), beam_size: int = 1, condition_lambda=1.0):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        self.conditioning_tokenizer = conditioning_tokenizer
        self.conditioning_model = conditioning_model
        self.model = model
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = 1
        self.inputs = inputs
        self.beam_size = beam_size
        self.condition_lambda = condition_lambda

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        topk_res = torch.topk(scores, top_k)

        # merge input_ids to token_ids
        new_tokens = self.model.tokenizer.batch_decode(topk_res.indices.reshape(-1), skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
        new_prefixes = self.model.tokenizer.batch_decode(input_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
        premises = []
        for i, prefix in enumerate(new_prefixes):
            for j in range(top_k):
                premises.append(prefix + new_tokens[i * top_k + j])

        # run conditional model
        cond_logits = self._run_conditional_model(premises)

        # merge logits
        cond_logits = cond_logits.reshape((-1, top_k)) * self.condition_lambda  # HERE condition_lambda=1.0
        words_in_premise = len(premises[0].strip().split(" "))
        if self.linear_warmup is True and words_in_premise < 6:
            if words_in_premise < 2:
                cond_logits = cond_logits * 0.0
            else:
                cond_logits = cond_logits * ((words_in_premise - 1) / 5.)
        cond_logits = cond_logits.type(scores.dtype)  # needed for 8bit
        scores = scores.scatter(1, topk_res.indices, cond_logits, reduce='add')
        return scores


class GetTrainingDataCriticLogitsWarper(GeneralCriticLogitsWarper):
    def __init__(self, model, conditioning_model, conditioning_tokenizer, inputs, top_k: int,
                 filter_value: float = -float("Inf"), beam_size=1, condition_lambda=1.0):
        super().__init__(model, conditioning_model, conditioning_tokenizer, inputs, top_k, filter_value, beam_size,
                         condition_lambda)
        self.results = []

    def _run_conditional_model(self, premises):
        hypothesies = list(
            itertools.chain.from_iterable(itertools.repeat(i, self.top_k * self.beam_size) for i in self.inputs))
        for p, h in zip(premises, hypothesies):
            self.results.append((p, h))
        return torch.ones(len(hypothesies), device="cuda")


class ClassifierCriticLogitsWarper(GeneralCriticLogitsWarper):
    """
    String Matcher
    """

    def __init__(self, model, conditioning_model, conditioning_tokenizer, inputs, top_k: int,
                 filter_value: float = -float("Inf"), beam_size=1, condition_lambda=1.0):
        super().__init__(model, conditioning_model, conditioning_tokenizer, inputs, top_k, filter_value, beam_size,
                         condition_lambda)

    def _run_conditional_model(self, premises):
        hypothesies = list(
            itertools.chain.from_iterable(itertools.repeat(i, self.top_k * self.beam_size) for i in self.inputs))
        cond_features1 = self.conditioning_tokenizer(hypothesies, premises, return_tensors='pt', truncation=True,
                                                     padding=True).to("cuda")
        cond_logits = self.conditioning_model._forward(cond_features1)
        return cond_logits


class CriticAwareInferenceModule(InferenceModule):
    def __init__(self, args, conditioning_model, conditioning_tokenizer, model_path=None, wrapper="nli"):
        super().__init__(args, model_path=model_path, training_module_cls=Seq2SeqTrainingModule)
        self.conditioning_model = conditioning_model
        self.conditioning_tokenizer = conditioning_tokenizer
        self.condition_lambda = args.condition_lambda
        self.beam_size = args.beam_size
        self.top_k = args.critic_top_k
        if wrapper == "classifier":
            self.wrapper = ClassifierCriticLogitsWarper
        elif wrapper == "data":
            self.wrapper = GetTrainingDataCriticLogitsWarper
        else:
            logger.error("Wrong wrapper")

    def predict_step(self, batch, batch_idx):
        in_sentences = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True)
        condition = self.wrapper(top_k=self.top_k, model=self.model, conditioning_model=self.conditioning_model,
                                 conditioning_tokenizer=self.conditioning_tokenizer, inputs=in_sentences,
                                 beam_size=self.beam_size, condition_lambda=self.condition_lambda)
        condition.linear_warmup = self.args.linear_warmup

        processors = LogitsProcessorList()
        processors.append(TopKLogitsWarper(top_k=self.top_k))
        processors.append(condition)

        model_out = self.model.model.generate(
            batch["input_ids"],
            max_length=self.args.max_length,
            num_beams=self.beam_size,
            num_return_sequences=1,
            logits_processor=processors
        )

        tokenizer_out = self.tokenizer.batch_decode(
            model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.info(tokenizer_out[0])
        if self.args.wrapper == "data":
            return {self.model.MODEL_OUT: model_out, self.model.TOKENIZER_OUT: tokenizer_out, "data": condition.results}

        return {self.model.MODEL_OUT: model_out, self.model.TOKENIZER_OUT: tokenizer_out}


class CriticGenDataInferenceModule(InferenceModule):
    def __init__(self, args, conditioning_model, conditioning_tokenizer, model_path=None, wrapper="nli"):
        super().__init__(args, model_path=model_path, training_module_cls=Seq2SeqTrainingModule)
        self.conditioning_model = conditioning_model
        self.conditioning_tokenizer = conditioning_tokenizer
        self.condition_lambda = args.condition_lambda
        self.beam_size = args.beam_size
        self.top_k = args.critic_top_k
        if wrapper == "classifier":
            self.wrapper = ClassifierCriticLogitsWarper
        elif wrapper == "data":
            self.wrapper = GetTrainingDataCriticLogitsWarper
        else:
            logger.error("Unknown wrapper")

    def predict_step(self, batch, batch_idx):
        in_sentences = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True)
        condition = self.wrapper(top_k=self.top_k, model=self.model, conditioning_model=self.conditioning_model,
                                 conditioning_tokenizer=self.conditioning_tokenizer, inputs=in_sentences,
                                 beam_size=self.beam_size, condition_lambda=self.condition_lambda)
        out_gold_sentences = batch["labels"]
        start_tokens = torch.ones(out_gold_sentences.shape[0], 1,
                                  dtype=int).cuda() * self.model.model.config.decoder_start_token_id
        out_gold_sentences = torch.column_stack((start_tokens, out_gold_sentences))
        ##NOT CLEAN
        condition.linear_warmup = self.args.linear_warmup

        processors = LogitsProcessorList()
        processors.append(TopKLogitsWarper(top_k=self.top_k))
        processors.append(condition)
        results = []
        for i in range(out_gold_sentences.shape[1]):
            model_out = self.model.model(input_ids=batch["input_ids"],
                                         decoder_input_ids=out_gold_sentences[:, :(i + 1)])
            probs = model_out.logits.softmax(dim=2)
            values, predictions = probs.topk(5)

            for k in range(predictions.shape[2]):
                preds = torch.column_stack((out_gold_sentences[:, :(i + 1)], predictions[:, -1, k]))
                preds_tokens = self.tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)
                for j, pred in enumerate(preds_tokens):
                    results.append((pred, in_sentences[j]))

        tokenizer_out = self.tokenizer.batch_decode(
            out_gold_sentences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        logger.info(tokenizer_out[0])
        if self.args.wrapper == "data":
            return {"data": set(results)}

        return {self.model.MODEL_OUT: model_out, self.model.TOKENIZER_OUT: tokenizer_out}
