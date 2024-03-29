import envs
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import (
    NormalizeObservation,
    NormalizeReward,
    RunningMeanStd,
)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from pytorch_lightning import LightningModule, Trainer
from collections import namedtuple
from base64 import b64encode

BatchItem = namedtuple("BatchItem", ("obs_b", "action_b", "return_b"))

# device = "cpu"
# accelarator = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accelarator = "gpu" if torch.cuda.is_available() else "cpu"
num_gpus = torch.cuda.device_count()


def create_env(env_name: str, num_envs: int) -> NormalizeReward:
    env = gym.vector.make(env_name, num_envs=num_envs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    return env


def plot_policy(policy):
    pos = np.linspace(-4.8, 4.8, 100)
    vel = np.random.random(size=(10000, 1)) * 0.1
    ang = np.linspace(-0.418, 0.418, 100)
    ang_vel = np.random.random(size=(10000, 1)) * 0.1

    g1, g2 = np.meshgrid(pos, ang)
    grid = np.stack((g1, g2), axis=-1)
    grid = grid.reshape(-1, 2)
    grid = np.hstack((grid, vel, ang_vel))

    probs = (
        policy(grid).detach().numpy()
        if device == "cpu"
        else policy(grid).detach().cpu().numpy()
    )
    probs_left = probs[:, 0]

    probs_left = probs_left.reshape(100, 100)
    probs_left = np.flip(probs_left, axis=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(probs_left, cmap="coolwarm")
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("P(left | s)", size=20)
    plt.xlabel("Cart Position", size=14)
    plt.ylabel("Pole angle", size=14)
    plt.xticks(ticks=[0, 50, 100], labels=["-4.8", "0", "4.8"])
    plt.yticks(ticks=[100, 50, 0], labels=["-0.418", "0", "0.418"])


def test_env(env_name: str, policy: nn.Module, obs_rms: RunningMeanStd):
    env = gym.make(env_name)
    env = RecordVideo(env, "videos", episode_trigger=lambda e: True)
    env = NormalizeObservation(env)
    env.obs_rms = obs_rms

    for episode in range(10):
        done = False
        obs = env.reset()
        while not done:
            action = policy(obs).multinomial(1).cpu().item()
            obs, _, done, _, _ = env.step(action)
    del env


# def display_video(episode=0):
#     video_file = open(f"/content/videos/rl-video-episode-{episode}.mp4", "r+b").read()
#     video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
#     return HTML(f"<video width=600 controls><source src='{video_url}'></video>")


class RLDataset(IterableDataset):
    def __init__(
        self,
        env: NormalizeReward,
        policy: nn.Module,
        steps_per_epoch: int,
        gamma: float,
    ):
        self.env = env
        self.policy = policy
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.obs = env.reset()

    @torch.no_grad()
    def __iter__(self):
        transitions = []

        for step in range(self.steps_per_epoch):
            action = self.policy(self.obs).to(device)
            action = action.multinomial(1).cpu().numpy()
            next_obs, reward, done, _, _ = self.env.step(action.flatten())
            transitions.append((self.obs, action, reward, done))
            self.obs = next_obs

        obs_b, action_b, reward_b, done_b = map(np.stack, zip(*transitions))

        running_return = np.zeros(self.env.num_envs, dtype=np.float32)
        return_b = np.zeros_like(reward_b)

        for row in range(self.steps_per_epoch - 1, -1, -1):
            running_return = (
                reward_b[row] + (1 - done_b[row]) * self.gamma * running_return
            )
            return_b[row] = running_return

        num_samples = self.env.num_envs * self.steps_per_epoch
        obs_b = obs_b.reshape(num_samples, -1)
        action_b = action_b.reshape(num_samples, -1)
        return_b = return_b.reshape(num_samples, -1)

        idx = list(range(num_samples))
        random.shuffle(idx)

        for i in idx:
            yield BatchItem(obs_b[i], action_b[i], return_b[i])


class GradientPolicy(nn.Module):
    def __init__(self, in_features: int, n_actions: int, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, n_actions).to(device)

    def forward(self, x: tuple[np.ndarray, dict]) -> torch.Tensor:
        t = torch.tensor(x[0]).float().to(device)
        t = F.relu(self.fc1(t)).to(device)
        t = F.relu(self.fc2(t)).to(device)
        t = F.softmax(self.fc3(t), dim=-1).to(device)
        return t


class Reinforce(LightningModule):
    def __init__(
        self,
        env_name: str,
        num_envs=8,
        samples_per_epoch=1000,
        batch_size=1024,
        hidden_size=64,
        policy_lr=0.001,
        gamma=0.99,
        entropy_coef=0.001,
        optim=AdamW,
        num_workers=0,
    ):

        super().__init__()

        self.env = create_env(env_name, num_envs=num_envs)

        obs_size: int = (
            self.env.unwrapped.single_observation_space.shape[0]
            if hasattr(self.env.unwrapped.single_observation_space, "shape")
            else len(self.env.unwrapped.single_observation_space)
        )
        n_actions: int = self.env.unwrapped.single_action_space.n  # type: ignore

        self.policy = GradientPolicy(obs_size, n_actions, hidden_size)
        self.dataset = RLDataset(self.env, self.policy, samples_per_epoch, gamma)
        self.num_workers = num_workers

        self.save_hyperparameters()

    # Configure optimizers.
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)  # type: ignore

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, pin_memory=True)  # type: ignore

    # Training step.
    def training_step(self, batch: BatchItem, batch_idx: int) -> float:
        obs, actions, returns = batch

        probs = self.policy(obs)
        log_probs = torch.log(probs + 1e-6)
        action_log_prob = log_probs.gather(1, actions)

        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        pg_loss = -action_log_prob * returns
        loss = (pg_loss - self.hparams.entropy_coef * entropy).mean()  # type: ignore

        self.log("episode/PG Loss", pg_loss.mean())
        self.log("episode/Entropy", entropy.mean())

        return loss

    def on_train_epoch_end(self, training_step_outputs):
        self.log("episode/Return", self.env.return_queue[-1])


# algo2 = Reinforce("CartPole-v1", num_workers=2)
# trainer2 = Trainer(
#     accelerator=accelarator, devices=num_gpus, max_epochs=100, log_every_n_steps=1
# )
# trainer2.fit(algo2)

algo = Reinforce("crypto-envs/BTCUSDT-v0").to(device)
trainer = Trainer(
    accelerator=accelarator, devices=num_gpus, max_epochs=100, log_every_n_steps=1
)
trainer.fit(algo)

import warnings

warnings.filterwarnings("ignore")

test_env("crypto-envs/BTCUSDT-v0", algo.policy.to(device), algo.env.obs_rms)
