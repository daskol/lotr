#!/usr/bin/env python

import logging
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from json import dumps
from pathlib import Path
from typing import Optional

import numpy as np
import torch as T
from datasets import load_dataset
from evaluate import Metric
from evaluate import load as load_metric
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoTokenizer, RobertaForSequenceClassification,
                          Trainer, TrainingArguments)
from transformers.integrations import TensorBoardCallback

from linear import LoTR, LoTRLinear
from util import map_module

RE_CORRECTION = re.compile(
    r'/roberta/encoder/layer/\d+/attention/self/(value|query)/correction/.*')

RE_HEAD = re.compile(r'/classifier/.*')

RE_PATTERNS = (RE_CORRECTION, RE_HEAD)

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('task')
parser.add_argument('--enable-lotr', action='store_true',
                    help='use LoTR factorization')

opt_iters = parser.add_argument_group('iteration options')
opt_iters.add_argument('--batch-size', default=16, type=int)
opt_iters.add_argument('--num-epoches', default=1, type=int)

opt_optim = parser.add_argument_group('optimizer options')
opt_optim.add_argument('--init-rank', default=1, type=int)
opt_optim.add_argument('--lr', default=2e-5, type=float,
                       help='learning rate (tau)')


def to_json(val) -> str:
    if isinstance(val, Path):
        return str(val)
    else:
        raise TypeError(f'No rule to serialize {type(val)} to JSON.')


def convert_lotr(module: T.nn.Module, path: str, lotr: LoTR):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return LoTRLinear.from_linear(module, lotr)


def convert_model(model, lotr: LoTR):
    return map_module(
        model,
        partial(convert_lotr, lotr=lotr),
        r'/roberta/encoder/layer/\d+/attention/self/(value|query)')


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


def compute_metrics(metric: Metric, inputs):
    predictions, references = inputs
    if metric.config_name != 'stsb':
        predictions = predictions.argmax(axis=1)
    else:
        predictions = predictions[..., 0]
    return metric.compute(predictions=predictions, references=references)


def train(task: str, batch_size=16, num_epoches=1, enable_lotr=False, rank=1,
          lr=2e-5, log_dir: Optional[Path] = None):
    logging.info('training options: %s',
                 dumps(locals(), ensure_ascii=False, default=to_json))

    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.info('load and initialize tokenizer')
    tokenizer_path = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize(inp):
        out = tokenizer(inp['sentence'],
                        max_length=256,
                        padding='max_length',
                        truncation=True,
                        return_tensors='np')
        out['idx'] = inp['idx']
        out['label'] = np.array(inp['label'])
        return out

    dataset = load_dataset('glue', task) \
        .map(tokenize, batched=True, remove_columns=['sentence']) \
        .with_format('numpy')

    logger.info('load model from zoo')
    model_path = 'roberta-base'
    model = RobertaForSequenceClassification \
        .from_pretrained(model_path) \
        .requires_grad_(False)
    if enable_lotr:
        logger.info('patch model to add LoTR term')
        lotr = LoTR(model.config.hidden_size, model.config.hidden_size, rank)
        model = convert_model(model, lotr)
        model = mask_weights(model)
        model.classifier.requires_grad_(True)
        for layer in model.roberta.encoder.layer:
            layer.attention.self.query.lotr.requires_grad_()
            layer.attention.self.value.lotr.requires_grad_()

    logging.info('common block:\n%s', model.roberta.encoder.layer[0])
    logging.info('number of parameters: total=%d trainable=%d ratio=%.2f%%',
                 model.num_parameters(),
                 model.num_parameters(True),
                 model.num_parameters(True) / model.num_parameters() * 100)

    if T.cuda.is_available():
        logger.info('move model to CUDA device')
        model = model.to('cuda')

    logger.info('load evaluation metrics')
    metric = load_metric('glue', task)

    logger.info('instantiate trainer')
    noepoches = 10
    warmup_steps = int(0.06 * len(dataset['train']) * noepoches / batch_size)
    model_dir = Path('model')

    args = TrainingArguments(
        output_dir=str(model_dir / f'glue-{task}'),
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoches,
        save_strategy='no',
        logging_strategy='epoch',
        log_level='warning',
        learning_rate=lr,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        lr_scheduler_type='polynomial',
        warmup_steps=warmup_steps,
        push_to_hub=False,
    )

    callbacks = []
    if log_dir is not None:
        tensorboard = SummaryWriter(log_dir)
        tensorboard_callback = TensorBoardCallback(tensorboard)
        callbacks.append(tensorboard_callback)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=partial(compute_metrics, metric),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info('evaluate metrics before training')
    eval_metrics = trainer.evaluate()
    logger.info('metrics: %s', eval_metrics)

    logger.info('enter training loop: noepoches=%d', num_epoches)
    trainer.train()


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    train(args.task, args.batch_size, args.num_epoches, args.enable_lotr,
          args.init_rank, args.lr)


if __name__ == '__main__':
    main(parser.parse_args())
