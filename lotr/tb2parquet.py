#!/usr/bin/env python3
"""Convert TensorBoard log files in a tree on a filesystem to a single
parquet-formatted file.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet

from lotr.tb import glob_combiner, read_scalars

RE_METRIC = (r'(eval|train)/'
             r'(accuracy|epoch|f1|learning_rate|loss|matthews_correlation'
             r'|pearson|spearmanr|train_loss)')


parser = ArgumentParser(description=__doc__)
parser.add_argument('log_dir', type=Path, help='log tree to convert')
parser.add_argument('output', type=Path, help='path to output parquet file')


def read_logs(log_dir: Path, pattern: str) -> pd.DataFrame:
    names = ['task', 'lr', 'rank', 'seed']
    with glob_combiner('*/*/*/*/*.tfevents.*', names) as combiner:
        frame = combiner(read_scalars, log_dir, pattern)
    frame['lr'] = frame.lr.astype(float)
    frame['rank'] = frame['rank'].astype(int)
    frame['seed'] = frame.seed.astype(int)
    return frame \
        .set_index(names + ['tag', 'step']) \
        .sort_index()


def compress_logs(log_dir: Path, output: Path):
    output_dir = output.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    frame = read_logs(log_dir, RE_METRIC)
    pa.parquet.write_table(pa.table(frame), output, compression='zstd',
                           compression_level=13)


def main(ns: Namespace):
    compress_logs(ns.log_dir, ns.output)


if __name__ == '__main__':
    main(parser.parse_args())
