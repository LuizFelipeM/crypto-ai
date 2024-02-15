from itertools import count
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from functools import reduce
from ai_models.depp_q_network import DQN
from environment import Environment, Space, current_pnl
from graphs import plot_reinforcement_learning
from utils import diminishing_return
from ._environmentManager import EnvironmentManager
from ._strategy import EpsilonGreedyStrategy
from ._agent import Agent
from ._replayMemory import ReplayMemory, Experience
from ._qvalues import QValues


env = Environment(
    Space(
        {
            # Comprar
            0: (
                lambda old_steps, _: diminishing_return(
                    reduce(
                        lambda prev, step: prev + 1 if step[1] == 0 else 0,
                        old_steps,
                        0,
                    ),
                    0.5,  # Improve with better calculation
                ),
                False,
            ),
            # Manter
            1: (current_pnl, False),
            # Vender
            2: (current_pnl, True),
        }
    ),
    pd.read_csv("./datasets/BTCUSDT - 1s - 2023-01 - 2023-12/BTCUSDT-1s-2023-01.csv"),
)


def extract_tensors(
    experiences: list[Experience],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


# Hiper parameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

# Environment and network setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = EnvironmentManager(device, env)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

(x, y) = em.get_shape()
policy_net = DQN(x, y).to(device)
target_net = DQN(x, y).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


def train_and_save_qnn(path: str) -> None:
    # Training
    episode_durations = []
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()

        for timestamp in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if em.done:
                episode_durations.append(timestamp)
                plot_reinforcement_learning(episode_durations, 100)
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net, f"{path}/policy_model.pth")
    torch.save(policy_net.state_dict(), f"{path}/policy_model_parameters.pth")
    torch.save(target_net, f"{path}/target_model.pth")
    torch.save(target_net.state_dict(), f"{path}/target_model_parameters.pth")
