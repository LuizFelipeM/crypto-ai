from datetime import datetime
import numpy as np
import tensorflow as tf
from ai_models import generate_model, save_model
from datasets import get_close_scaler, get_train_test_dataset
from graphs import plot_prediction
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


folder_path = (
    f"./models/BTCUSDT-1m-2023-10/{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"
)

X_train, X_test, y_train, y_test, labels = get_train_test_dataset()


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
    es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=100)
    input.model.fit(
        input.X_train,
        input.y_train,
        epochs=5000,
        batch_size=X_train.shape[0],
        callbacks=[es],
    )
    save_model(input.folder_path, input.model_name, input.model)
    return Output(input.folder_path, input.model_name, input.model)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs_to_train = [
        Input(
            folder_path,
            "LSTM_D_LSTM",
            generate_model(
                LSTM(
                    units=100,
                    return_sequences=True,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                ),
                Dropout(0.2),
                Dense(units=50),
                Dropout(0.2),
                LSTM(units=100),
                Dense(units=1, activation="sigmoid"),
            ),
            X_train,
            y_train,
        )
    ]

    for input in inputs_to_train:
        input.model.compile(optimizer="adam", loss="mean_squared_error")

    subset_factor = 500
    for input in inputs_to_train:
        output = train_and_save_model(input)
        y_predicted = output.model.predict(X_test)
        plot_prediction(
            output.folder_path,
            output.model_name,
            get_close_scaler(),
            labels.tolist()[::subset_factor],
            y_test[::subset_factor],
            y_predicted[::subset_factor],
        )
