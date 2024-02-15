import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.preprocessing import MinMaxScaler
from utils import get_moving_average


def plot_reinforcement_learning(values: list, moving_avg_period: int) -> None:
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)


def plot_prediction(
    folder_path: str,
    file_name: str,
    scaler: MinMaxScaler,
    x_labels: Sequence[str],
    y_true,
    y_pred,
) -> None:
    fig = plt.figure()
    res_true = scaler.inverse_transform(y_true)
    res_pred = scaler.inverse_transform(y_pred)
    plt.plot(np.reshape(res_true, (-1,)), color="red", label="Real close price")
    plt.plot(np.reshape(res_pred, (-1,)), color="blue", label="Predicted close price")
    plt.title("BTCUSDT close prices")
    plt.xlabel("Time")
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=30)  # type: ignore
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"{folder_path}/{file_name}_graph.png")
    plt.close(fig)


def plot_candlesticks(df: pd.DataFrame, datetime_format="%Y-%m-%d %H:%M:%S"):
    with plt.ion():
        plt.figure()

        up = df[df.close >= df.open]
        down = df[df.close < df.open]

        down_color = "red"
        upper_color = "green"

        width = 0.5
        width2 = 0.05

        plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=upper_color)
        plt.bar(
            up.index, up.high - up.close, width2, bottom=up.close, color=upper_color
        )
        plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=upper_color)

        plt.bar(
            down.index,
            down.close - down.open,
            width,
            bottom=down.open,
            color=down_color,
        )
        plt.bar(
            down.index,
            down.high - down.open,
            width2,
            bottom=down.open,
            color=down_color,
        )
        plt.bar(
            down.index,
            down.low - down.close,
            width2,
            bottom=down.close,
            color=down_color,
        )

        plt.xlabel("Time")
        plt.ylabel("USDT")

        close_time = pd.to_datetime(
            df.close_time, unit="ms", origin="unix"
        ).dt.strftime(datetime_format)
        plt.xticks(ticks=range(len(close_time)), labels=close_time, rotation=30)  # type: ignore
        plt.grid()
        plt.show()
