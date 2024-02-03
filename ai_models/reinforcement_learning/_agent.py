import random
import torch
import torch.nn as nn
from ai_models.reinforcement_learning._strategy import Strategy


class Agent:
    def __init__(
        self, strategy: Strategy, num_actions: int, device: torch.device
    ) -> None:
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net: nn.Module) -> torch.Tensor:
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # Explore environment
        else:
            with torch.no_grad():
                return (
                    policy_net(state).argmax(dim=1).to(self.device)
                )  # Exploit environment
