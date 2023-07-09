#!/usr/bin/env python3

import logging
from argparse import Namespace
from itertools import product
from pathlib import Path
from random import shuffle

from lotr_glue import parser, train

LRS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
RANKS = [1, 2, 5, 10, 16, 32, 64, 128, 256, 512]

parser.add_argument('--log-dir', default=Path('log'), type=Path)


def main(args: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    if not args.enable_lotr:
        raise NotImplementedError('Search only LoTR hyperparameters.')

    search_space = [*product(LRS, RANKS)]
    shuffle(search_space)  # Want to explore more but not thorough.
    logging.info('search space size is %d', len(search_space))

    for it, (lr, rank) in enumerate(search_space):
        logging.info('attempt #%02d: lr=%e, rank=%d', it, lr, rank)
        log_dir = args.log_dir / 'lotr' / f'{lr:e}' / str(rank)
        train(task=args.task,
              batch_size=args.batch_size,
              num_epoches=args.num_epoches,
              enable_lotr=args.enable_lotr,
              rank=rank,
              lr=lr,
              log_dir=log_dir)


if __name__ == '__main__':
    main(parser.parse_args())
