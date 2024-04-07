import math
import gymnasium as gym
import pandas as pd
import polars as pl
import numpy as np
import pygame
from functools import reduce
from typing import Any, List, Literal, Optional, SupportsFloat, Union, Tuple, overload
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Box
from numpy.typing import NDArray
from gymnasium.core import RenderFrame
from utils import diminishing_return, pnl, Position, Contract


class BTCUSDTEnv(gym.Env):
    # metadata = {"render_modes": ["human"], "render_fps": 1}
    # window: pygame.surface.Surface | None
    # clock: pygame.time.Clock | None

    _data_frame: pl.LazyFrame
    _actions: List[int] = []
    _start_timestep = 0
    _max_timestep = 2676455

    @property
    def _current_timestep(self) -> int:
        return self._start_timestep + len(self._actions)

    def __init__(self, render_mode: str | None = None, timeframe=600) -> None:
        self.timeframe = timeframe
        self.window_size = 512
        self._data_frame = pl.scan_csv(
            "./datasets/BTCUSDT - 1s - 2023-01 - 2023-12/BTCUSDT-1s-2023-01.csv"
        )

        self.action_space = Discrete(3)
        # self.observation_space = Dict(
        #     {
        #         "open_time": Discrete(1_000_000_000_000),
        #         "open": Box(low=0, high=math.inf, shape=()),
        #         "high": Box(low=0, high=math.inf, shape=()),
        #         "low": Box(low=0, high=math.inf, shape=()),
        #         "close": Box(low=0, high=math.inf, shape=()),
        #         "volume": Box(low=0, high=math.inf, shape=()),
        #         "close_time": Discrete(1_000_000_000_000),
        #         "quote_volume": Box(low=0, high=math.inf, shape=()),
        #         "count": Discrete(1_000_000_000_000),
        #         "taker_buy_volume": Box(low=0, high=math.inf, shape=()),
        #         "taker_buy_quote_volume": Box(low=0, high=math.inf, shape=()),
        #         "ignore": Discrete(1),
        #     }
        # )

        # self.observation_space = Tuple(
        #     (
        #         Discrete(1_000_000_000_000),
        #         Box(low=0, high=math.inf, shape=()),
        #         Box(low=0, high=math.inf, shape=()),
        #         Box(low=0, high=math.inf, shape=()),
        #         Box(low=0, high=math.inf, shape=()),
        #         Box(low=0, high=math.inf, shape=()),
        #         Discrete(1_000_000_000_000),
        #         Box(low=0, high=math.inf, shape=()),
        #         Discrete(1_000_000_000_000),
        #         Box(low=0, high=math.inf, shape=()),
        #         Box(low=0, high=math.inf, shape=()),
        #         Discrete(1),
        #     )
        # )

        self.observation_space = Box(low=0, high=math.inf, shape=(12,))

        self._action_to_reward = {
            # Buy
            0: (self._buy, False),
            # Hold
            1: (self._current_pnl, False),
            # Sell
            2: (self._current_pnl, True),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[NDArray[Any], dict]:
        super().reset(seed=seed, options=options)

        self._actions = []
        self._start_timestep = self.np_random.integers(
            0, high=self._max_timestep - self.timeframe, dtype=int
        )

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        self._actions.append(action)
        rewarder, terminated = self._action_to_reward[action]

        reward = rewarder()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, np.array([True]), info

    # def render(self) -> RenderFrame | list[RenderFrame] | None:
    #     return self._render_frame()

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_current_as_data_frame(self) -> pl.DataFrame:
        return self._data_frame.slice(self._current_timestep, 1).collect()

    def _get_current_as_tuple(self) -> tuple[Any, ...]:
        return self._get_current_as_data_frame().row()

    def _get_current_as_dict(self) -> dict[str, Any]:
        return self._get_current_as_data_frame().to_dict()

    def _get_obs(self) -> np.ndarray:
        return self._get_current_as_data_frame().to_numpy()

    def _get_info(self) -> dict:
        curr_pnl = self._current_pnl()
        curr_roi = self._get_roi(self._get_actions_idxs(0), curr_pnl)
        return {
            "pnl": curr_pnl,
            "roi": curr_roi,
            "candle": self._get_current_as_dict(),
        }

    def _get_pnl(self, open_idxs: List[int], close_price: float) -> float:
        return reduce(
            lambda prev, idx: prev
            + pnl(
                Position.LONG,
                Contract.LINEAR,
                self._data_frame.slice(idx, 1).collect().item(0, "open"),
                close_price,
                position_qty=0.01,
            ),
            open_idxs,
            0.0,
        )

    def _get_roi(self, open_idxs: List[int], pnl: float) -> float:
        if not any(open_idxs):
            return 0

        return pnl / reduce(
            lambda prev, idx: prev
            + self._data_frame.slice(idx, 1).collect().item(0, "open"),
            open_idxs,
            0.0,
        )

    def _get_actions_idxs(self, action: int) -> List[int]:
        return [i + 1 for i, a in enumerate(self._actions) if a == action]

    def _current_pnl(self) -> float:
        return self._get_pnl(
            self._get_actions_idxs(0),
            self._data_frame.slice(self._current_timestep, 1)
            .collect()
            .item(0, "close"),
        )

    def _buy(self) -> float:
        return diminishing_return(
            len(self._get_actions_idxs(0)),
            0.5,  # Improve with better calculation
        )

    def _render_frame(self) -> RenderFrame | List[RenderFrame] | None:
        pass

    # Fix when implementing the human render_mode
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels

    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )

    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )

    #     if (
    #         self.render_mode == "human"
    #         and self.window is not None
    #         and self.clock is pygame.time.Clock
    #     ):
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()

    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )
