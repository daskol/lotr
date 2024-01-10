#!/usr/bin/env python3
"""Convert TensorBoard log files in a tree on a filesystem to a single
parquet-formatted file.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet

from lotr.tb import read_scalars, rglob_combiner

RE_METRIC = (r'(eval|train)/'
             r'(accuracy|epoch|f1|learning_rate|loss|matthews_correlation'
             r'|pearson|spearmanr|train_loss)')


parser = ArgumentParser(description=__doc__)
parser.add_argument('-i', '--index', type=str, nargs='*',
                    help='names of index columns')
parser.add_argument('-n', '--name', type=str, nargs='*',
                    help='names of path segments')
parser.add_argument('log_dir', type=Path, help='log tree to convert')
parser.add_argument('output', type=Path, help='path to output parquet file')


def read_logs(log_dir: Path, pattern: str,
              names: Optional[list[str]] = None,
              index: Optional[list[str]] = None) -> pd.DataFrame:
    with rglob_combiner(names) as combiner:
        frame = combiner(read_scalars, log_dir, pattern)
    # If index columns are not specified then use default index. Column `value`
    # (from tfevent) is kept as a value column. While `tag` and `step` columns
    # are appended to the end of list of index columns. If there is a `seed`
    # column then it moved at the end of the list.
    if not index:
        index = frame.columns.to_list()
        index.remove('value')  # Non-index column.
        index.remove('tag')
        index.remove('step')
        index.extend(['tag', 'step'])
        try:
            index.remove('seed')
            index.append('seed')
        except ValueError:
            pass
    return frame \
        .set_index(index) \
        .sort_index()


def compress_logs(log_dir: Path, output: Path,
                  names: Optional[list[str]] = None,
                  index: Optional[list[str]] = None):
    # Read tensorboard event files to data frame.
    frame = read_logs(log_dir, RE_METRIC, names, index)

    # Convert data frame to table and write it to parque file.
    output_dir = output.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    pa.parquet.write_table(pa.table(frame), output, compression='zstd',
                           compression_level=13)


def main(ns: Namespace):
    compress_logs(ns.log_dir, ns.output, ns.name, ns.index)


if __name__ == '__main__':
    main(parser.parse_args())
