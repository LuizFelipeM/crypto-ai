from abc import ABC, abstractmethod
import math


class Strategy(ABC):
    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay

    @abstractmethod
    def get_exploration_rate(self, current_step: int) -> float:
        raise NotImplementedError


class EpsilonGreedyStrategy(Strategy):
    def get_exploration_rate(self, current_step: int) -> float:
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )
