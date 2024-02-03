from functools import reduce
from environment._environment import (
    Environment,
    Observation,
    Reward,
    Step,
)
from environment._space import Space
from utils import Contract, Position, pnl

__all__ = ["Environment", "Space"]


def current_pnl(old_steps: list[Step], current_observation: Observation) -> Reward:
    return reduce(
        lambda prev, step: prev
        + pnl(
            Position.LONG,
            Contract.LINEAR,
            step[0].open,
            current_observation.close,
            position_qty=0.01,
        ),
        filter(lambda step: step[1] == 0, old_steps),
        0.0,
    )
