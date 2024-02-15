from envs.btcusdt import BTCUSDTEnv
from gymnasium.envs.registration import register

__all__ = ["BTCUSDTEnv"]

register(
    id="crypto-envs/BTCUSDT-v0",
    entry_point="envs:BTCUSDTEnv",
    max_episode_steps=500,
)
