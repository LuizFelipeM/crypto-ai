import os
from typing import Sequence
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from database import DbContext, MySqlConfig
from database.repositories import KlineRepository
from sklearn.preprocessing import MinMaxScaler
from utils import split_X_y

sc = MinMaxScaler(feature_range=(0, 1))
close_sc = MinMaxScaler(feature_range=(0, 1))

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

X_test = np.concatenate((X_train[len(X_train) - padding :], X_test), axis=0)  # type: ignore
y_test = np.concatenate((y_train[len(y_train) - padding :], y_test), axis=0).reshape(
    (-1, 1)
)


def get_train_test_dataset() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    labels = selected_df.index[-len(y_test) :].values
    return (X_train, X_test, y_train, y_test, labels)


def get_close_scaler() -> MinMaxScaler:
    return close_sc
