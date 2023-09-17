#!/usr/bin/env python3
"""In this experiment we are trying to find a better scaling factor before
additive correction in LoTR.

    ./lotr_glue_rank.py \
            --enable-lotr \
            --log-dir log/scaler/inverse-rank \
            --num-epoches 20 \
            cola
"""

from argparse import Namespace
from functools import partial
from pathlib import Path
import logging

from lotr_glue import make_trainer, parser, TASKS
import numpy as np
from optuna import Trial, create_study
from optuna.samplers import GridSampler

parser.add_argument('--log-dir', default=Path('log'), type=Path)
parser.add_argument('task', default='cola', choices=TASKS)

SEARCH_SPACE = {
    'lr': [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3],
    'rank': [1, 2, 5, 10, 16, 32, 64, 128, 256, 512],
    'seed': [42, 3705, 0x12c946425095e587],
}


def objective(args: Namespace, trial: Trial):
    lr = trial.suggest_categorical('lr', SEARCH_SPACE['lr'])
    rank = trial.suggest_categorical('rank', SEARCH_SPACE['rank'])
    seed = trial.suggest_categorical('seed', SEARCH_SPACE['seed'])
    task = args.task
    logging.info('run experiment: rank=%d lr=%e seed=%d', rank, lr, seed)
    log_dir = args.log_dir / 'lotr' / task / f'{lr:e}' / str(rank) / str(seed)
    trainer = make_trainer(task=task,
                           batch_size=args.batch_size,
                           num_epoches=args.num_epoches,
                           enable_lotr=args.enable_lotr,
                           rank=rank,
                           lr=lr,
                           log_dir=log_dir,
                           seed=seed)
    result = trainer.train()
    return result.training_loss


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    if not args.enable_lotr:
        raise NotImplementedError('Search only LoTR hyperparameters.')

    space_size = np.product([len(v) for v in SEARCH_SPACE.values()])
    logging.info('search space size is %d', space_size)

    task = args.task
    study = create_study(study_name=f'scaler/inverse-rank/{task}',
                         storage='sqlite:///iclr24.sqlite',
                         sampler=GridSampler(SEARCH_SPACE),
                         load_if_exists=True)
    study.optimize(partial(objective, args), n_trials=space_size)


if __name__ == '__main__':
    main(parser.parse_args())
