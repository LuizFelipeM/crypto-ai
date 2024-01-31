from collections import namedtuple
from typing import Callable, TypeAlias
from ai_models.reinforcement_learning.Environment.Space import Space

Action: TypeAlias = int
Reward: TypeAlias = float
Terminated: TypeAlias = bool
Step: TypeAlias = tuple[Action, Reward]
StepResult: TypeAlias = tuple[float, Terminated, bool]
ActionResults: TypeAlias = tuple[Callable[[list[Step]], Reward], Terminated]


class Environment:
    action_space: Space[Action, ActionResults]
    observation_space: 
    _previous_steps: list[Step] = []


    def __init__(self, action_space: Space[Action, ActionResults]) -> None:
        self.action_space = action_space

    def step(self, action: Action) -> StepResult:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(reward, terminated, truncated)`.

        Args:
            action (int): an action provided by the agent

        Returns:
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
        """
        rewarder, terminated = self.action_space[action]
        reward = rewarder(self._previous_steps)
        self._previous_steps.append((action, reward))
        return (reward, terminated, False)

    def reset(self) -> None:
        self._previous_steps = []
