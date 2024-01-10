"""Module tensorboard provides a small set of routines for data extraction out
of tensorboard summary data.
"""

import re
from contextlib import contextmanager
from itertools import count, zip_longest
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import pandas as pd
from google.protobuf.message import Message
from tensorboard import data_compat, dataclass_compat
from tensorboard.compat.proto import types_pb2
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub import errors, pywrap_tensorflow

__all__ = ('glob_combiner', 'read_events', 'read_messages', 'read_messages',
           'rglob_combiner')

ACCESSORS = {
    types_pb2.DT_DOUBLE: attrgetter('double_val'),
    types_pb2.DT_FLOAT: attrgetter('float_val'),
    types_pb2.DT_INT16: attrgetter('int_val'),
    types_pb2.DT_INT32: attrgetter('int_val'),
    types_pb2.DT_INT64: attrgetter('int64_val'),
    types_pb2.DT_INT8: attrgetter('int_val'),
    types_pb2.DT_UINT16: attrgetter('int_val'),
    types_pb2.DT_UINT32: attrgetter('int_val'),
    types_pb2.DT_UINT32: attrgetter('uint32_val'),
    types_pb2.DT_UINT64: attrgetter('uint64_val'),
    types_pb2.DT_UINT8: attrgetter('int_val'),
}


def is_int(value) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_float(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def infer_dtype(value) -> np.dtype:
    match (is_float(value), is_int(value)):
        case (_, True):
            return int
        case (True, False):
            return float
        case (False, False):
            return pd.CategoricalDtype(categories=[value])


def infer_dtypes(values) -> tuple[np.dtype, ...]:
    return tuple([infer_dtype(value) for value in values])


@contextmanager
def glob_combiner(pattern, names: Optional[list[str]] = None,
                  args=None, kwargs=None):
    """
    >>> glob = '*/*/*.tfevents.*'
    >>> names = ['task', 'alpha']
    >>> path = 'log/baseline/nag4'
    >>> pattern = 'train/loss'
    >>> with glob_combiner(glob, names) as combiner:
    >>>     frame = combiner(read_scalars, path, pattern)
    """

    def name(names: Optional[list[str]] = None):
        for lhs, rhs in zip_longest(names or [], count()):
            yield str(rhs) if lhs is None else lhs

    def combiner(reader, base_dir, *args, **kwargs):
        base_dir = Path(base_dir)
        frames = []
        for path in base_dir.glob(pattern):
            # Extract path segments from relative path to base directory. Then
            # assign as much as possible names to each path segment. Otherwise,
            # assign ordinal number of a segment.
            parts = path.parent.relative_to(base_dir).parts
            col_names, col_values = zip(*zip(name(names), parts))
            # Read to file to dataframe and append columns about path.
            frame = reader(path, *args, **kwargs)
            frame[list(col_names)] = list(col_values)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    yield combiner


@contextmanager
def rglob_combiner(names: Optional[list[str]] = None, args=None, kwargs=None):
    """High-order function which reads all tfevents-files in directory tree.

    >>> names = ['task', 'alpha']
    >>> with glob_combiner(names) as combiner:
    >>>     frame = combiner(read_scalars, 'log/glue', 'train/loss')
    """

    def as_name(names: Optional[list[str]] = None):
        for lhs, rhs in zip_longest(names or [], count()):
            yield str(rhs) if lhs is None else lhs

    def combiner(reader, base_dir, *args, **kwargs):
        base_dir = Path(base_dir)
        frames = []
        for path in base_dir.rglob('*.tfevents.*'):
            # Extract path segments from relative path to base directory. Then
            # assign as much as possible names to each path segment. Otherwise,
            # assign ordinal number of a segment.
            parts = path.parent.relative_to(base_dir).parts

            # Read to file to dataframe and append columns about path.
            frame = reader(path, *args, **kwargs)

            col_names, col_values = zip(*zip(as_name(names), parts))
            col_dtypes = infer_dtypes(parts)
            frame[list(col_names)] = list(col_values)
            for name, dtype in zip(col_names, col_dtypes):
                frame[name] = frame[name].astype(dtype)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    yield combiner


def read_messages(path: PathLike, message_type: Type[Message]) -> Message:
    reader = pywrap_tensorflow.PyRecordReader_New(str(path))
    while True:
        try:
            reader.GetNext()
        except errors.OutOfRangeError:
            return
        string = reader.record()
        record = message_type.FromString(string)
        yield record


def read_events(path: PathLike, metadata: dict[str, Any] = {}) -> Event:
    metadata = metadata or {}  # Container in arguments is mutable.
    for event in read_messages(path, Event):
        event = data_compat.migrate_event(event)
        events = dataclass_compat.migrate_event(event, metadata)
        yield from events


def read_scalars(path: PathLike, pattern: str | re.Pattern) -> pd.DataFrame:
    regexp = re.compile(pattern)

    col_step = []
    col_tag = []
    col_value = []

    for event in read_events(path):
        if event.summary is None:
            continue

        for value in event.summary.value:
            if not regexp.match(value.tag):
                continue

            if (value_ty := value.WhichOneof('value')) is None:
                continue
            elif value_ty == 'simple_scalar':
                col_step.append(event.step)
                col_tag.append(value.tag)
                col_value.append(value.simple_scalar)
            elif value_ty == 'tensor':
                if len(value.tensor.tensor_shape.dim) != 0:
                    continue
                if (getter := ACCESSORS.get(value.tensor.dtype)) is None:
                    continue
                col_step.append(event.step)
                col_tag.append(value.tag)
                col_value.append(getter(value.tensor)[0])
            else:
                # We does not support other types which are not scalar for now.
                continue

    col_step = np.array(col_step, dtype=int)  # Enforce type conversion.
    return pd.DataFrame({'step': col_step, 'tag': col_tag, 'value': col_value})
