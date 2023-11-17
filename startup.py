import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing import Pool
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sqlalchemy import ScalarResult
from database import DbContext, MySqlConfig
from database.repositories import KlineRepository
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def get_available_devices():
    local_device_protos = tf.config.list_physical_devices()
    return [x.name for x in local_device_protos]


load_dotenv(override=False)

print(f"list_physical_devices = {tf.config.list_physical_devices('GPU')}")

print(f"get_available_devices = {get_available_devices()}")

# mysql_config = MySqlConfig(os.environ.get("MySql:ConnectionString"))

# dbc = DbContext(mysql_config)
# kr = KlineRepository(dbc)
# selected_fields = [
#     "id",
#     "open_time",
#     "close_time",
#     "symbol",
#     "interval",
#     "open_price",
#     "close_price",
#     "high_price",
#     "low_price",
#     "base_asset_volume",
#     "number_of_trades",
#     "is_kline_closed",
#     "quote_asset_volume",
#     "taker_buy_base_asset_volume",
#     "taker_buy_quote_asset_volume",
#     "created_at",
# ]


# def create_dataframe(values: ScalarResult) -> pd.DataFrame:
#     df = pd.DataFrame([{f: getattr(r, f) for f in selected_fields} for r in values])
#     df.shape
#     return df


# def create_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1, activation="sigmoid"))

#     model.compile(optimizer="adam", loss="mean_squared_error")
#     es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=58)
#     model.fit(X_train, y_train, epochs=500, batch_size=28, callbacks=[es])
#     return model


# def split_X_y(
#     dataset_shape: tuple[int, int], training_set: np.ndarray, padding: int
# ) -> tuple[list, list]:
#     X = []
#     y = []

#     for i in range(padding, dataset_shape[0]):
#         X.append(training_set[i - padding : i, 0])
#         y.append(training_set[i, 0])

#     X, y = np.array(X), np.array(y)
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#     return (X, y)


# # kline_batch = kr.batch(842)
# # df = create_dataframe(kline_batch.next())
# df = create_dataframe(kr.get_all())

# close_prices = df["close_price"].values.reshape(-1, 1)

# sc = MinMaxScaler(feature_range=(0, 1))
# close_prices_scaled = sc.fit_transform(close_prices)

# padding = 140
# X, y = split_X_y(df.shape, close_prices_scaled, padding)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=0
# )

# model = create_model(X_train, y_train)

# X_test = np.concatenate((X_train[len(X_train) - padding :], X_test), axis=0)
# y_test = np.concatenate((y_train[len(y_train) - padding :], y_test), axis=0).reshape(
#     (-1, 1)
# )
# y_predict = model.predict(X_test)

# plt.plot(sc.inverse_transform(y_test), color="red", label="Real close price")
# plt.plot(sc.inverse_transform(y_predict), color="blue", label="Predicted close price")
# plt.title("BTCUSDT close prices")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend()
# plt.savefig("./graph.png")


# with Pool(processes=3) as pool:
#     results = pool.imap_unordered()
