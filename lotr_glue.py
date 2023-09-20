#!/usr/bin/env python

import logging
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from json import dumps
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import torch as T
from datasets import load_dataset
from evaluate import Metric
from evaluate import load as load_metric
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoTokenizer, RobertaForSequenceClassification,
                          Trainer, TrainingArguments,
                          get_polynomial_decay_schedule_with_warmup)
from transformers.integrations import TensorBoardCallback

from low_rank import (LowRankAdamW, LowRankLinear, LRSchedulerList,
                      OptimizerList)
from util import map_module

RE_PARAM_LOW_RANK = re.compile(r'roberta\.encoder\.layer\.\d+\.'
                               r'attention\.self\.(value|query)\.correction')

RE_PARAM_CLASSIFIER = re.compile(r'classifier\..*$')

TASK_TO_HYPERPARAMS = {
    'cola': (16, 1e-5),
    'mnli': (16, 1e-5),
    'mnli-mm': (16, 1e-5),
    'mrpc': (16, 1e-5),
    'qnli': (32, 1e-5),  # NOTE We used batch size 16 instead of 32.
    'qqp': (32, 1e-5),
    'rte': (16, 2e-5),
    'sst2': (32, 2e-5),
    'stsb': (16, 1e-5),
    'wnli': (32, 1e-5),
}

TASK_TO_KEYS = {
    'cola': ('sentence', ),
    'mnli': ('premise', 'hypothesis'),
    'mnli-mm': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', ),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

TASKS = [*sorted(TASK_TO_KEYS.keys())]

RE_CORRECTION = re.compile(
    r'/roberta/encoder/layer/\d+/attention/self/(value|query)/correction/.*')

RE_HEAD = re.compile(r'/classifier/.*')

RE_PATTERNS = (RE_CORRECTION, RE_HEAD)

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('--enable-lotr', action='store_true',
                    help='use LoTR factorization')

opt_iters = parser.add_argument_group('iteration options')
opt_iters.add_argument('--batch-size', default=16, type=int)
opt_iters.add_argument('--num-epoches', default=1, type=int)

opt_optim = parser.add_argument_group('optimizer options')
opt_optim.add_argument('--rank', default=1, type=int)
opt_optim.add_argument('--lr', default=2e-5, type=float,
                       help='learning rate (tau)')


def to_json(val) -> str:
    if isinstance(val, Path):
        return str(val)
    else:
        raise TypeError(f'No rule to serialize {type(val)} to JSON.')


def convert_low_rank(module: T.nn.Module, path: str):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return LowRankLinear.from_linear(module)


def convert_model(model):
    return map_module(
        model, partial(convert_low_rank),
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


def instantiate_model(model_path='roberta-base', enable_lotr=True,
                      num_labels=2) -> RobertaForSequenceClassification:
    model = RobertaForSequenceClassification \
        .from_pretrained(model_path, num_labels=num_labels) \
        .requires_grad_(False)
    if enable_lotr:
        # logger.info('patch model to add correction term')
        model = convert_model(model)
        model = mask_weights(model)
        model.classifier.requires_grad_(True)
        for layer in model.roberta.encoder.layer:
            layer.attention.self.query.correction.requires_grad_()
            layer.attention.self.value.correction.requires_grad_()
    return model


def instantiate_optimizer(model, lr, rank, batch_size, num_epoches,
                          num_samples):

    def filter_params(pattern):
        return [(name, param) for name, param in model.named_parameters()
                if param.requires_grad and pattern.match(name)]

    def print_params(named_params, label=None):
        if label is not None:
            print(f'{label}:', end=' ')
        print('total', len(named_params), 'parameters:')
        names = [k for k, v in named_params]
        if len(names) > 5:
            names = names[:4] + ['...']
        pprint(names)

    def slice_params(named_params):
        return (v for k, v in named_params)

    named_params_cls = filter_params(RE_PARAM_CLASSIFIER)
    named_params_lr = filter_params(RE_PARAM_LOW_RANK)
    params_cls = slice_params(named_params_cls)
    params_lr = slice_params(named_params_lr)
    print_params(named_params_cls)
    print_params(named_params_lr)

    opt_kwargs = {
        'lr': lr,
        'betas': (0.9, 0.98),
        'eps': 1e-6,
        'weight_decay': 0.1
    }

    opt_cls = T.optim.AdamW(params_cls, **opt_kwargs)
    opt_lr = LowRankAdamW(params_lr, rank=rank, **opt_kwargs)
    opt = OptimizerList([opt_cls, opt_lr])

    # In reference fine-tuning report for RoBERTa on GLUE.
    noepoches = 10
    warmup_steps = int(0.06 * num_samples * noepoches / batch_size)
    total_steps = num_samples * num_epoches / batch_size

    def make_scheduler(opt: T.optim.Optimizer):
        return get_polynomial_decay_schedule_with_warmup(
            opt, warmup_steps, total_steps)

    scheduler_cls = make_scheduler(opt_cls)
    scheduler_lr = make_scheduler(opt_lr)
    scheduler = LRSchedulerList([scheduler_cls, scheduler_lr])

    return opt, scheduler


def make_trainer(task: str, batch_size=16, num_epoches=1, enable_lotr=False,
                 rank=1, lr=2e-5, log_dir: Optional[Path] = None,
                 seed: Optional[int] = None):
    logging.info('training options: %s',
                 dumps(locals(), ensure_ascii=False, default=to_json))

    if seed is not None:
        T.manual_seed(seed)

    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.info('load and initialize tokenizer')
    tokenizer_path = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_keys = TASK_TO_KEYS[task]

    def tokenize(inp):
        args = [inp[key] for key in tokenizer_keys]
        out = tokenizer(*args,
                        max_length=256,
                        padding='max_length',
                        truncation=True,
                        return_tensors='np')
        out['idx'] = inp['idx']
        out['label'] = np.array(inp['label'])
        return out

    dataset = load_dataset('glue', task) \
        .map(tokenize, batched=True, remove_columns=tokenizer_keys) \
        .with_format('numpy')

    # Load and configure model output head for a specific GLUE task.
    if task in ('mnli', 'mnli-mm'):
        num_labels = 3
    elif task == 'stsb':
        num_labels = 1
    else:
        num_labels = 2

    logger.info('load model from zoo')
    model_path = 'roberta-base'
    model = instantiate_model(model_path, enable_lotr, num_labels)

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

    logger.info('instantiate a list of optimizers and schedulers')
    noepoches = 10
    num_samples = len(dataset['train'])
    opt, scheduler = instantiate_optimizer(model, lr, rank, batch_size,
                                           num_epoches, num_samples)

    logger.info('instantiate trainer')
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
        push_to_hub=False,
    )

    callbacks = []
    if log_dir is not None:
        tensorboard = SummaryWriter(log_dir)
        tensorboard_callback = TensorBoardCallback(tensorboard)
        callbacks.append(tensorboard_callback)

    return Trainer(
        model=model,
        args=args,
        optimizers=(opt, scheduler),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=partial(compute_metrics, metric),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )


def train(task: str, batch_size=16, num_epoches=1, enable_lotr=False, rank=1,
          lr=2e-5, log_dir: Optional[Path] = None,
          seed: Optional[int] = None):
    trainer = make_trainer(task, batch_size, num_epoches, enable_lotr, rank,
                           lr, log_dir, seed)
    logger.info('evaluate metrics before training')
    eval_metrics = trainer.evaluate()
    logger.info('metrics: %s', eval_metrics)
    logger.info('enter training loop: noepoches=%d', num_epoches)
    return trainer.train()


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    train(args.task, args.batch_size, args.num_epoches, args.enable_lotr,
          args.rank, args.lr)


if __name__ == '__main__':
    parser.add_argument('task')  # Add task argument to a generic parser.
    main(parser.parse_args())
