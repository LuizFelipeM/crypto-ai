import random
from typing import Callable, TypeAlias
from pandas import DataFrame, Series
from environment._space import Space

Observation: TypeAlias = Series
Action: TypeAlias = int
Reward: TypeAlias = float
Terminated: TypeAlias = bool
Step: TypeAlias = tuple[Observation, Action, Reward]
StepResult: TypeAlias = tuple[Observation, float, Terminated, bool]
ActionResults: TypeAlias = tuple[
    Callable[[list[Step], Observation], Reward], Terminated
]


class Environment:
    _previous_steps: list[Step] = []
    _gap: int
    current_time_frame: int = 0
    action_space: Space[Action, ActionResults]
    observation_space: DataFrame

    def __init__(
        self, action_space: Space[Action, ActionResults], env_space: DataFrame, gap=100
    ) -> None:
        self.action_space = action_space
        self.observation_space = env_space
        self._gap = gap
        self.reset()

    @property
    def current_observation_space(self) -> Series:
        return self.observation_space.iloc[self.current_time_frame]

    def step(self, action: Action) -> StepResult:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(reward, terminated, truncated)`.

        Args:
            action (int): an action provided by the agent

        Returns:
            observation (Observation): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
        """
        self.current_time_frame += 1
        rewarder, terminated = self.action_space[action]
        reward = rewarder(
            self._previous_steps, self.current_observation_space
        )  # Corrigir, está enviando apenas os steps anteriores sem o step atual (impedindo calculo de PnL ao manter ou vender)
        truncated = self.current_time_frame >= self.observation_space.shape[0]
        observation = Series() if truncated else self.current_observation_space
        self._previous_steps.append((observation, action, reward))
        return (
            observation,
            reward,
            terminated,
            truncated,
        )

    def reset(self) -> None:
        self.current_time_frame = random.randint(
            0, self.observation_space.shape[0] - self._gap
        )
        self._previous_steps = []
