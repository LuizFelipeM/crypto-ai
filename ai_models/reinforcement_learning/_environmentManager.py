import torch
from environment import Environment


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
        _, reward, self.done, _ = self.env.step(action.item())  # type: ignore
        return torch.tensor([reward], device=self.device)

    def get_state(self) -> torch.Tensor:
        return torch.from_numpy(self.env.current_observation_space.to_numpy())

    def get_shape(self) -> tuple[int, int]:
        return self.env.observation_space.shape
