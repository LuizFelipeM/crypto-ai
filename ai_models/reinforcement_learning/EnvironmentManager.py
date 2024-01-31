import torch
from ai_models.reinforcement_learning.Environment.Environment import Environment


class EnvironmentManager:
    device: torch.device
    env: Environment
    done: bool
    is_starting: bool

    def __init__(self, device: torch.device, env: Environment) -> None:
        self.done = False
        self.device = device
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.env.reset()
        self.is_starting = True

    def num_actions_available(self) -> int:
        return self.env.action_space.n

    def take_action(self, action: torch.Tensor) -> torch.Tensor:
        _, reward, self.done, _, _ = self.env.step(action.item())  # type: ignore
        return torch.tensor([reward], device=self.device)

    def get_state(self) -> torch.Tensor:
        raise NotImplementedError
