import os
from datetime import datetime
from typing import Any, Generic, TypeVar
from keras.models import Model
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from numpy import ndarray
from pandas import DataFrame


# T = TypeVar("T")


class LSTMModel:
    model = Sequential()
    es: EarlyStopping

    def __init__(self, shape: tuple[int, int]) -> None:
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation="sigmoid"))

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=30)

    def fit(
        self, X_train: DataFrame, y_train: DataFrame, epochs: int, batch_size: int
    ) -> None:
        self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[self.es]
        )

    def predict(self, X) -> Any:
        return self.model.predict(X)


def save_model(folder_path: str, folder_name: str, model: Model) -> None:
    folder_path = f"{folder_path}/{folder_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.save(f"{folder_path}/model.h5")

    model_json = model.to_json()
    with open(f"{folder_path}/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{folder_path}/model_weights.h5")


def create_LSTM_model(train_shape: tuple[int, ...]) -> Sequential:
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            # input_shape=(X_train.shape[1], X_train.shape[2]),
            input_shape=(train_shape[1], train_shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def create_RNN_model(train_shape: tuple[int, ...]) -> Sequential:
    model = Sequential()
    model.add(
        SimpleRNN(
            units=50,
            # return_sequences=True,
            # input_shape=(X_train.shape[1], X_train.shape[2]),
            input_shape=(train_shape[1], train_shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def early_stopping_train_model(
    model: Model, X_train: ndarray, y_train: ndarray
) -> Model:
    es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=58)
    model.fit(X_train, y_train, epochs=500, batch_size=X_train.shape[0], callbacks=[es])
    return model
