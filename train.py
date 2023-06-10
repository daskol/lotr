#!/usr/bin/env python

import logging
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from json import dumps
from os import getenv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch as T
from torch.optim import AdamW
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

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
parser.add_argument('--enable-kls', action='store_true',
                    help='use KLS-optimizer')

opt_iters = parser.add_argument_group('iteration options')
opt_iters.add_argument('--batch-size', default=8, type=int)
opt_iters.add_argument('--num-epoches', default=1, type=int)

opt_optim = parser.add_argument_group('optimizer options')
opt_optim.add_argument('--init-rank', default=1, type=int)
opt_optim.add_argument('--lr', default=2e-5, type=float,
                       help='learning rate (tau)')
opt_optim.add_argument('--rank-tol', default=0.1, type=float,
                       help='rank truncation threshold (theta)')


def convert_lora_linear(module: T.nn.Module, path: str, rank, adaptive):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return AdaptiveLoRaLinear.from_linear(module, rank=rank, adaptive=adaptive)


def convert_dlrt_linear(module: T.nn.Module, path: str, rank, adaptive):
    if not isinstance(module, T.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    # Enforce use of non-adaptive KLS-factorized linear layer for output
    # module (classification problem).
    if module.out_features == 2:
        adaptive = False
        rank = 2
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


def set_step_mode(model: T.nn.Module, mode: Mode):
    # Set step modes for adaptive low-rank linear layers.
    for layer in model.layer:
        if isinstance(layer, AdaptiveLoRaLinear):
            layer.mode = mode
    # Set step mode for KLS-layers directly.
    model.classifier.dense.step = mode.name
    model.classifier.out_proj.step = mode.name


def apply_fn(batch: Dict[str, Any], model, mode: Mode = Mode.S,
             enable_kls: bool = False) -> SequenceClassifierOutput:
    """Apply model to input batch in specific step mode for KLS-optimizer if
    KLS is enabled.
    """
    if enable_kls:
        # Change computation mode in forward pass.
        if mode in (Mode.K, Mode.L, Mode.S):
            set_step_mode(model, mode)
        else:
            raise ValueError(f'Unknown step mode: {mode}.')

    # Apply model to input batch.
    return model(input_ids=batch['input_ids'],
                 attention_mask=batch['attention_mask'],
                 labels=batch['label'])


def evaluate(metric, dataset, model, enable_kls=False):
    model.eval()
    preds = []
    refs = []
    batches = dataset \
        .with_format('torch', device=model.device) \
        .iter(batch_size=8)
    for batch in batches:
        with T.no_grad():
            output = apply_fn(batch, model, enable_kls=enable_kls)
        preds.append(output.logits.argmax(-1))
        refs.append(batch['label'])
    return metric.compute(predictions=T.hstack(preds),
                          references=T.hstack(refs))


def train(task: str, batch_size=8, num_epoches=1, enable_kls=False, rank=1,
          adaptive=True, lr=2e-5, rank_tol=0.1):
    logging.info('training options: %s', dumps(locals()))

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
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    if enable_kls:
        logger.info('patch model for KLS-optimizer')
        model = convert_model(model, rank, adaptive)
        model = mask_weights(model)
        model.layer = [*flatten_module(model).values()]

    # TODO
    from transformers import TrainingArguments, Trainer
    num_train_steps = len(dataset['train']) * num_epoches // batch_size
    num_warmup_steps = int(0.06 * num_train_steps)
    args = TrainingArguments(output_dir='tmp',
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
                             warmup_steps=num_warmup_steps,
                             push_to_hub=False)

    metric = load_metric('glue', task)
    trainer = Trainer(
        model=model.to('cuda'),
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=lambda x: metric.compute(
            predictions=x.predictions.argmax(-1), references=x.label_ids))
    trainer.train()
    exit()

    if T.cuda.is_available():
        logger.info('move model to CUDA device')
        model = model.to('cuda')

    if enable_kls:
        logger.info('initialize DLRT-optimizer')
        opt = KLSOptimizer(model)
    else:
        logger.info('initialize AdamW-optimizer')
        num_train_steps = len(dataset['train']) * num_epoches // batch_size
        num_warmup_steps = int(0.06 * num_train_steps)
        from transformers import get_polynomial_decay_schedule_with_warmup
        opt = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps)

    logger.info('load evaluation metrics')
    metric = load_metric('glue', task)
    eval_fn = partial(evaluate, metric, dataset['validation'],
                      enable_kls=enable_kls)
    eval_metrics = eval_fn(model)
    logger.info('metrics: %s', eval_metrics)

    logger.info('initialize random number generator')
    bits = np.random.MT19937(42)
    prng = np.random.Generator(bits)

    logger.info('enter training loop: noepoches=%d', num_epoches)
    for epoch in range(num_epoches):
        logger.info('epoch #%02d begins', epoch)
        batches = dataset['train'] \
            .shuffle(generator=prng,
                     keep_in_memory=True,
                     load_from_cache_file=False) \
            .with_format('torch', device=model.device) \
            .iter(batch_size=batch_size, drop_last_batch=True)

        if getenv('INTERACTIVE', 'false').lower() == 'true':
            from tqdm import tqdm
            nobatches = len(dataset['train']) // batch_size
            batches = tqdm(batches, total=nobatches, unit='batch')

        model.train()
        for it, batch in enumerate(batches):
            def loss_fn(mode: Mode = Mode.S):
                output = apply_fn(batch, model, mode, enable_kls)
                return output.loss

            if enable_kls:
                # Make preparation step.
                model.zero_grad()
                opt.zero_grad()
                opt.preprocess_step()
                # Make K-step.
                loss = loss_fn(Mode.K)
                loss.backward()
                # Make L-step.
                loss = loss_fn(Mode.L)
                loss.backward()
                # Make S-step.
                opt.step(loss_fn)
            else:
                loss = loss_fn()
                loss.backward()
                opt.step()
                scheduler.step()

            if it % 2:
                logger.info('[%02d:%03d] train/loss=%e', epoch, it,
                            loss.detach().item())

        logger.info('[%0d] evaluate model', epoch + 1)
        eval_metrics = eval_fn(model)
        logger.info('[%0d] eval metrics: %s', epoch + 1, eval_metrics)


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    train(args.task, args.batch_size, args.num_epoches, args.enable_kls,
          args.init_rank, True, args.lr, args.rank_tol)


if __name__ == '__main__':
    main(parser.parse_args())
