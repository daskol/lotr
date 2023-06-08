#!/usr/bin/env python

import logging
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch as T
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification

from linear import AdaptiveLoRaLinear, DLRTLinear, Mode
from optim import KLSOptimizer
from util import flatten_module, map_module

RE_CORRECTION = re.compile(
    r'/roberta/encoder/layer/\d+/attention/self/(value|query)/correction/.*')

RE_HEAD = re.compile(r'/classifier/.*')

RE_PATTERNS = (RE_CORRECTION, RE_HEAD)

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('task')


def convert_lora_linear(module: T.nn.Module, path: str, rank, adaptive):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return AdaptiveLoRaLinear.from_linear(module, rank=rank, adaptive=adaptive)


def convert_dlrt_linear(module: T.nn.Module, path: str, rank, adaptive):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    device = module.weight.device
    dtype = module.weight.dtype
    return DLRTLinear(module.in_features, module.out_features,
                      module.bias is not None, device, dtype, rank, adaptive)


def convert_model(model, rank=None, adaptive=True):
    model = map_module(
        model, partial(convert_lora_linear, rank=rank, adaptive=adaptive),
        r'/roberta/encoder/layer/\d+/attention/self/(value|query)')
    model = map_module(
        model, partial(convert_dlrt_linear, rank=rank, adaptive=adaptive),
        r'/classifier/(dense|out_proj)')
    return model


def mask_weights(model: T.nn.Module):
    """Mask trainable weights only in classification head and in KLS-correction
    in value and query of self-attention module.
    """
    for name, param in model.named_parameters():
        name = '/' + name.replace('.', '/')
        for pattern in RE_PATTERNS:
            if pattern.match(name) is not None:
                param.requires_grad = True
                break
        else:
            param.requires_grad = False
    return model


def train(task: str, batch_size=1, rank=1, adaptive=True):
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
    model = mask_weights(model)
    model.layer = [*flatten_module(model).values()]

    def set_step_mode(model: T.nn.Module, mode: Mode):
        # Set step modes for adaptive low-rank linear layers.
        for layer in model.layer:
            if isinstance(layer, AdaptiveLoRaLinear):
                layer.mode = mode
        # Set step mode for KLS-layers directly.
        model.classifier.dense.step = mode.name
        model.classifier.out_proj.step = mode.name

    def apply_fn(batch: Dict[str, Any], mode: Mode = Mode.S) -> T.Tensor:
        # Change computation mode in forward pass.
        if mode in (Mode.K, Mode.L, Mode.S):
            set_step_mode(model, mode)
        else:
            raise ValueError(f'Unknown step mode: {mode}.')

        # Apply model to input batch.
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['label'])
        return output.loss

    logger.info('initialize DLRT-optimizer')
    opt = KLSOptimizer(model)

    logger.info('initialize random number generator')
    bits = np.random.MT19937(42)
    prng = np.random.Generator(bits)

    noepoches = 1
    logger.info('enter training loop: noepoches=%d', noepoches)
    for epoch in range(noepoches):
        logger.info('epoch #%02d begins', epoch)
        batches = dataset['train'] \
            .shuffle(generator=prng,
                     keep_in_memory=True,
                     load_from_cache_file=False) \
            .with_format('torch') \
            .iter(batch_size=batch_size, drop_last_batch=True)
        for batch in batches:

            def loss_fn(mode: Mode = Mode.S):
                return apply_fn(batch, mode)

            # Make preparation step.
            model.zero_grad()
            opt.zero_grad()
            opt.preprocess_step()

            # Make K-step and L=step.
            loss = loss_fn(Mode.K)
            loss.backward()

            loss = loss_fn(Mode.L)
            loss.backward()

            # Make S-step.
            opt.step(loss_fn)
            break  # TODO(@bershatsky): Debug iteration.


def main(args: Namespace):
    train(args.task)


if __name__ == '__main__':
    main(parser.parse_args())
