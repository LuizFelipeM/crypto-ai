import pandas as pd
import torch
from functools import reduce
from environment import Environment, Space, current_pnl
from utils import diminishing_return
from reinforcement_learning._environmentManager import EnvironmentManager


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_manager = EnvironmentManager(device, env)
