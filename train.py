#!/usr/bin/env python

import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch as T
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification

from linear import AdaptiveLoRaLinear
from util import map_module

logger = logging.getLogger(__name__)


def convert_layer(module: T.nn.Module, path: str, rank, adaptive):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return AdaptiveLoRaLinear.from_linear(module, rank=rank, adaptive=adaptive)


def convert_model(model, rank=None, adaptive=True):
    return map_module(
        model, partial(convert_layer, rank=rank, adaptive=adaptive),
        r'/roberta/encoder/layer/\d+/attention/self/(value|query)')


def train(task: str, batch_size=1, rank=None, adaptive=True):
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.info('load and initialize tokenizer')
    tokenizer_path = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize(inp):
        out = tokenizer(inp['sentence'],
                        max_length=256,
                        padding=True,
                        truncation=True,
                        return_tensors='np')
        out['idx'] = inp['idx']
        out['label'] = np.array(inp['label'])
        return out

    dataset = load_dataset('glue', task) \
        .map(tokenize, batched=True, remove_columns=['sentence']) \
        .with_format('numpy')

    logger.info('load model from zoo and patch it')
    model_path = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model = convert_model(model, rank, adaptive)

    logger.info('initialize random number generator')
    bits = np.random.MT19937(42)
    prng = np.random.Generator(bits)

    noepoches = 1
    logger.info('enter training loop: noepoches=%d', noepoches)
    for epoch in range(noepoches):
        batches = dataset['train'] \
            .shuffle(generator=prng,
                     keep_in_memory=True,
                     load_from_cache_file=False) \
            .iter(batch_size=batch_size, drop_last_batch=True)
        for batch in batches:
            break  # TODO(@bershatsky): Make a step.
