from typing import Any, Generic, TypeVar
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout


# T = TypeVar("T")


class LSTMModel:
    model = Sequential()
    es: EarlyStopping

    def __init__(self) -> None:
        self.model.add(
            LSTM(
                units=50,
                return_sequences=True,
                #    input_shape=(, 1)
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation="sigmoid"))

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=30)

    def fit(self, X_train, y_train, epochs: int, batch_size: int) -> None:
        self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[self.es]
        )

    def predict(self, X) -> Any:
        return self.model.predict(X)
