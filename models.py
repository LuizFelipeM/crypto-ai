generate_model(
    SimpleRNN(
        units=50,
        # return_sequences=True,
        # input_shape=(X_train.shape[1], X_train.shape[2]),
        input_shape=(train_shape[1], train_shape[2]),
    ),
    Dropout(0.2),
    Dense(units=1, activation="sigmoid"),
)
