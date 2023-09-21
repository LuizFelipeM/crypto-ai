import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load historical Bitcoin price data (CSV format)
# Replace 'bitcoin_price.csv' with the path to your data file
data = pd.read_csv("BTC-2017-2021min.csv")
# unix,date,symbol,open,high,low,close,Volume BTC,Volume USD

# Define features (e.g., historical prices, trading volume, technical indicators)
# and target variable (price movement: 1 for increase, 0 for decrease)
x = data[["open", "high", "low", "Volume BTC"]].values
y = (data["close"].shift(-1) > data["close"]).astype(int).values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train a logistic regression model (you can use more sophisticated models)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict the price movement for a new data point
# new_data_point = np.array(
#     [[new_open, new_high, new_low, new_volume]]
# )  # Replace with actual data
# predicted_movement = model.predict(new_data_point)
# print(
#     f'Predicted Price Movement: {"Increase" if predicted_movement[0] == 1 else "Decrease"}'
# )
