import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from database import MySqlConfig, DbContext
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


def date_parser(epoch_date: int) -> str:
    time_in_secs = epoch_date / 1000
    return datetime.fromtimestamp(float(time_in_secs)).strftime(
        # "%Y-%m-%d"
        "%Y-%m-%d %H:%M"
    )


padding = 1440
df = pd.read_csv(
    # "./datasets/BTCUSDT - 1mo - 2021-01 - 2023-10/BTCUSDT-1mo-2021-01-2023-10.csv"
    # "./datasets/BTCUSDT - 1d - 2021-01 - 2023-10/BTCUSDT-1d-2021-01-2023-10.csv"
    "./datasets/BTCUSDT - 1m - 2021-01 - 2023-10/BTCUSDT-1m-2023-10.csv",
    # index_col=6,
)

# close_prices = df["close_price"].values.reshape(-1, 1)
selected_df = df[
    [
        "close_time",
        "close",
        "open",
        "high",
        "low",
        "volume",
        # "quote_volume",
        # "count",
        # "taker_buy_volume",
        # "taker_buy_quote_volume",
    ]
]
# selected_df.loc[:, "close_time"] = pd.to_datetime(
#     selected_df.close_time, unit="ms", origin="unix"
# ).dt.strftime(
#     "%Y-%m-%d"
#     # "%Y-%m-%d %H:%M"
# )

selected_df.loc[:, "close"] = close_sc.fit_transform(
    selected_df.loc[:, "close":"close"]
)

columns_to_transform = selected_df.columns[selected_df.columns != "close"]
selected_df[columns_to_transform] = sc.fit_transform(selected_df[columns_to_transform])

X, y = split_X_y(selected_df.shape, selected_df, padding)
X = [x.set_index("close_time") for x in X]
y = y.set_index("close_time")


X_values = np.array(list(map(lambda x: x.iloc[:].values, X)))
y_values = y.loc[:, "close"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.3, random_state=0
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
