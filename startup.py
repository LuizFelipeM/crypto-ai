from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from ai_models import (
    create_LSTM_model,
    create_RNN_model,
    early_stopping_train_model,
    save_model,
)
from database import DbContext, MySqlConfig
from database.repositories import KlineRepository
from sklearn.preprocessing import MinMaxScaler
from graphs import plot_prediction
from keras.models import Model
from utils import split_X_y


def get_available_devices():
    local_device_protos = tf.config.list_physical_devices()
    return [x.name for x in local_device_protos]


load_dotenv(override=False)

mysql_config = MySqlConfig(str(os.environ.get("MySql:ConnectionString")))

dbc = DbContext(mysql_config)
kr = KlineRepository(dbc)
selected_fields = [
    "id",
    "open_time",
    "close_time",
    "symbol",
    "interval",
    "open_price",
    "close_price",
    "high_price",
    "low_price",
    "base_asset_volume",
    "number_of_trades",
    "is_kline_closed",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "created_at",
]


sc = MinMaxScaler(feature_range=(0, 1))
close_sc = MinMaxScaler(feature_range=(0, 1))


# kline_batch = kr.batch(842)
# df = create_dataframe(kline_batch.next())
# df = create_dataframe(kr.get_all())

df = pd.read_csv(
    "./datasets/BTCUSDT - 1mo - 2021-01 - 2023-10/BTCUSDT-1mo-2021-01-2023-10.csv"
)

# close_prices = df["close_price"].values.reshape(-1, 1)
selected_df = df[
    [
        "close_time",
        "close",
        "open",
        "high",
        "low",
        "volume"
        # "quote_volume",
        # "count",
        # "taker_buy_volume",
        # "taker_buy_quote_volume",
    ]
]
selected_df.close_time = pd.to_datetime(
    selected_df.close_time, unit="ms", origin="unix"
).dt.strftime("%Y-%m-%d")
selected_df = selected_df.set_index("close_time")

columns_to_transform = selected_df.columns[selected_df.columns != "close"]
selected_df[columns_to_transform] = sc.fit_transform(selected_df[columns_to_transform])

selected_df.close = close_sc.fit_transform(selected_df.loc[:, "close":"close"])

padding = 12
X, y = split_X_y(selected_df.shape, selected_df, padding)

X_values = np.array(list(map(lambda x: x.iloc[:].values, X)))
y_values = y.loc[:, "close"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.25, random_state=0
)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train_array = np.array(y_train)


folder_path = f"./models/{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"

X_test = np.concatenate((X_train[len(X_train) - padding :], X_test), axis=0)  # type: ignore
y_test = np.concatenate((y_train[len(y_train) - padding :], y_test), axis=0).reshape(
    (-1, 1)
)


class Input:
    folder_path: str
    model_name: str
    model: Model
    X_train: np.ndarray
    y_train: np.ndarray

    def __init__(
        self,
        folder_path: str,
        model_name: str,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        self.folder_path = folder_path
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.y_train = y_train


class Output:
    folder_path: str
    model_name: str
    model: Model

    def __init__(self, folder_path: str, model_name: str, model: Model) -> None:
        self.folder_path = folder_path
        self.model_name = model_name
        self.model = model


def train_and_save_model(input: Input) -> Output:
    model = early_stopping_train_model(input.model, input.X_train, input.y_train)
    save_model(input.folder_path, input.model_name, model)
    return Output(input.folder_path, input.model_name, model)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs_to_train = [
        Input(folder_path, "LSTM", create_LSTM_model(X_train.shape), X_train, y_train),
        Input(folder_path, "RNN", create_RNN_model(X_train.shape), X_train, y_train),
    ]

    for input in inputs_to_train:
        input.model.compile(optimizer="adam", loss="mean_squared_error")

    for input in inputs_to_train:
        output = train_and_save_model(input)
        y_predicted = output.model.predict(X_test)
        plot_prediction(
            output.folder_path, output.model_name, close_sc, y_test, y_predicted
        )
