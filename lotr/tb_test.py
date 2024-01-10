from pathlib import Path

import pytest

from lotr.tb import read_scalars, rglob_combiner

root_dir = Path(__file__).parent.parent
log_dir = root_dir / 'log'


@pytest.mark.skipif(not (log_dir / 'glue').exists(), reason='No test data.')
def test_rglob_combiner():
    names = ['model', 'method', 'task', 'lr', 'rank', 'seed']
    with rglob_combiner(names) as combine:
        df = combine(read_scalars, log_dir / 'glue', 'train/loss')
    assert len(df) > 0
    columns = df[names]
    assert columns['model'].dtype == 'category'
    assert columns['method'].dtype == 'category'
    assert columns['task'].dtype == 'category'
    assert columns['lr'].dtype == float
    assert columns['rank'].dtype == int
    assert columns['seed'].dtype == int
