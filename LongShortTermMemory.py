import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Load historical Bitcoin price data (CSV format)
# Replace 'bitcoin_price.csv' with the path to your data file
data = pd.read_csv("BTC-2017-2021min.csv")
# unix,date,symbol,open,high,low,close,Volume BTC,Volume USD

# Preprocess the data
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)
data = data[["close"]]  # Use only the 'Close' price for simplicity

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for the LSTM model
sequence_length = 30  # Adjust as needed
x = []
y = []

for i in range(sequence_length, len(data_scaled)):
    x.append(data_scaled[i - sequence_length : i, 0])
    y.append(1 if data_scaled[i, 0] > data_scaled[i - 1, 0] else 0)

x, y = np.array(x), np.array(y)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(x) * split_ratio)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model
y_pred = (model.predict(x_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict the price movement for a new data point
# new_data_point = data_scaled[-sequence_length:].reshape(1, -1, 1)
# predicted_movement = (model.predict(new_data_point) > 0.5).astype(int)
# print(
#     f'Predicted Price Movement: {"Increase" if predicted_movement[0][0] == 1 else "Decrease"}'
# )

# # Plot the predictions (optional)
# y_pred_all = (model.predict(x) > 0.5).astype(int)
# plt.figure(figsize=(12, 6))
# plt.plot(
#     data.index[sequence_length:], y_pred_all, label="Predicted Movement", marker="o"
# )
# plt.xlabel("Date")
# plt.ylabel("Price Movement")
# plt.legend()
# plt.grid(True)
# plt.show()
