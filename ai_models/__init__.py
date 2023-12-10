import os
from keras.models import Model
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from keras.src.engine.base_layer import BaseRandomLayer
from numpy import ndarray


def generate_model(*layers: list[BaseRandomLayer]) -> Sequential:
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model


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
    return generate_model(
        LSTM(
            units=50,
            return_sequences=True,
            # input_shape=(X_train.shape[1], X_train.shape[2]),
            input_shape=(train_shape[1], train_shape[2]),
        ),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1, activation="sigmoid"),
    )


def create_RNN_model(train_shape: tuple[int, ...]) -> Sequential:
    return generate_model(
        SimpleRNN(
            units=50,
            # return_sequences=True,
            # input_shape=(X_train.shape[1], X_train.shape[2]),
            input_shape=(train_shape[1], train_shape[2]),
        ),
        Dropout(0.2),
        Dense(units=1, activation="sigmoid"),
    )
