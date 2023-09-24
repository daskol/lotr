"""Package `lotr` contains reference implementation of low tensor rank adapter
for parameter efficient fine-tuning of large neural networks as well as
implementation of concurrent approaches like LoRA.
"""

from lotr.lora import LoRALinear
from lotr.lotr import LoTR, LoTRLinear
from lotr.low_rank import LowRankLinear
from lotr.optim import (LoTRAdam, LoTRAdamW, LoTRMixin, LoTRSGD, LowRankAdam,
                        LowRankAdamW, LowRankMixin, LowRankSGD, LRScheduler,
                        LRSchedulerList, OptimizerList)
from lotr.util import map_module

__all__ = ('LoRALinear', 'LoTR', 'LoTRAdam', 'LoTRAdamW', 'LoTRLinear',
           'LoTRMixin', 'LoTRSGD', 'LowRankAdam', 'LowRankAdamW',
           'LowRankLinear', 'LowRankMixin', 'LowRankSGD', 'LRScheduler',
           'LRSchedulerList', 'OptimizerList', 'map_module')
