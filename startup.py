from datetime import datetime
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
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K


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


def create_dataframe(values: ScalarResult) -> pd.DataFrame:
    df = pd.DataFrame([{f: getattr(r, f) for f in selected_fields} for r in values])
    df.shape
    return df


def split_X_y(
    dataset_shape: tuple[int, int], training_set: pd.DataFrame, padding: int
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    X: list[pd.DataFrame] = []
    y: list[pd.DataFrame] = []

    for i in range(padding, dataset_shape[0]):
        X.append(training_set.iloc[i - padding : i])
        y.append(training_set.iloc[[i]])

    return (X, pd.concat(y))


def save_model(name: str, model: Sequential) -> str:
    folder_name = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"
    folder_path = f"./models/{folder_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.save(f"{folder_path}/model.h5")

    model_json = model.to_json()
    with open(f"{folder_path}/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{folder_path}/model_weights.h5")
    return folder_path


def plot_prediction(folder_path: str, scaler: MinMaxScaler, y_true, y_pred) -> None:
    fig = plt.figure()
    res_true = scaler.inverse_transform(y_true)
    res_pred = scaler.inverse_transform(y_pred)
    plt.plot(np.reshape(res_true, (-1,)), color="red", label="Real close price")
    plt.plot(np.reshape(res_pred, (-1,)), color="blue", label="Predicted close price")
    plt.title("BTCUSDT close prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"{folder_path}/graph.png")
    plt.close(fig)


def create_LSTM_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="mean_squared_error")
    es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=58)
    model.fit(X_train, y_train, epochs=500, batch_size=X_train.shape[0], callbacks=[es])
    return model


def create_RNN_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(
        SimpleRNN(
            units=50,
            # return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="mean_squared_error")
    es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=58)
    model.fit(X_train, y_train, epochs=500, batch_size=X_train.shape[0], callbacks=[es])
    return model


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
]  # .values.reshape(-1, 1)
selected_df.close_time = pd.to_datetime(
    selected_df.close_time, unit="ms", origin="unix"
).dt.strftime("%Y-%m-%d")
selected_df = selected_df.set_index("close_time")
# selected_df.index.name = "close_time"
# selected_df.close = sc.fit_transform(selected_df.loc[:, "close":"close"])

# selected_df[selected_df.columns] = sc.fit_transform(selected_df[selected_df.columns])

selected_df.close = close_sc.fit_transform(selected_df.loc[:, "close":"close"])

selected_df.open = sc.fit_transform(selected_df.loc[:, "open":"open"])
selected_df.high = sc.fit_transform(selected_df.loc[:, "high":"high"])
selected_df.low = sc.fit_transform(selected_df.loc[:, "low":"low"])
selected_df.volume = sc.fit_transform(selected_df.loc[:, "volume":"volume"])

padding = 12
X, y = split_X_y(selected_df.shape, selected_df, padding)

X_values = np.array(list(map(lambda x: x.iloc[:].values, X)))
y_values = y.loc[:, "close"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.25, random_state=0
)

X_train = np.array(X_train)
X_test = np.array(X_test)

# X, y = np.array(X), np.array(y)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

y_train_array = np.array(y_train)
lstm_model = create_LSTM_model(X_train, y_train_array)
rnn_model = create_RNN_model(X_train, y_train_array)

lstm_folder_path = save_model("LSTM", lstm_model)
rnn_folder_path = save_model("RNN", rnn_model)

X_test = np.concatenate((X_train[len(X_train) - padding :], X_test), axis=0)
y_test = np.concatenate((y_train[len(y_train) - padding :], y_test), axis=0).reshape(
    (-1, 1)
)

y_lstm_predict = lstm_model.predict(X_test)
y_rnn_predict = rnn_model.predict(X_test)

plot_prediction(lstm_folder_path, close_sc, y_test, y_lstm_predict)
plot_prediction(rnn_folder_path, close_sc, y_test, y_rnn_predict)

# with Pool(processes=3) as pool:
#     results = pool.imap_unordered()
