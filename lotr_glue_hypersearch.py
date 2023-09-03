#!/usr/bin/env python3

import logging
from argparse import Namespace
from functools import partial
from pathlib import Path

from optuna import Trial, create_study
from optuna.samplers import GridSampler

from lotr_glue import parser, train

parser.add_argument('--log-dir', default=Path('log'), type=Path)

SEARCH_SPACE = {
    'lr': [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3],
    'rank': [1, 2, 5, 10, 16, 32, 64, 128, 256, 512],
    'seed': [42, 3705, 0x12c946425095e587],
    'task': [
        'cola',
        'mrpc',
        'rte',
        'sst2',
        'stsb',
        'wnli',
    ],
}


def objective(args: Namespace, trial: Trial):
    lr = trial.suggest_categorical('lr', SEARCH_SPACE['lr'])
    rank = trial.suggest_categorical('rank', SEARCH_SPACE['rank'])
    seed = trial.suggest_categorical('seed', SEARCH_SPACE['seed'])
    task = trial.suggest_categorical('task', SEARCH_SPACE['task'])
    print(f'task={task} rank={rank} lr={lr} seed={seed}')
    log_dir = args.log_dir / 'lotr' / task / f'{lr:e}' / str(rank) / str(seed)
    output = train(task=task,
                   batch_size=args.batch_size,
                   num_epoches=args.num_epoches,
                   enable_lotr=args.enable_lotr,
                   rank=rank,
                   lr=lr,
                   log_dir=log_dir,
                   seed=seed)
    return output.training_loss


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    if not args.enable_lotr:
        raise NotImplementedError('Search only LoTR hyperparameters.')

    study = create_study(study_name='roberta/glue/grid-search/small',
                         storage='sqlite:///iclr24.sqlite',
                         sampler=GridSampler(SEARCH_SPACE),
                         load_if_exists=True)
    study.optimize(partial(objective, args), n_trials=200)


if __name__ == '__main__':
    main(parser.parse_args())
