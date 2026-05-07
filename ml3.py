from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Auto MPG dataset
dataset = load_dataset("scikit-learn/auto-mpg")
df = dataset["train"].to_pandas()

# Inspect columns
print(df.columns)

# Drop the car name column if it is in the dataset
df = df.drop(columns=["name", "car name", "car_name"], errors="ignore")

# Replace question marks with missing values
df = df.replace("?", np.nan)

# Convert all columns to numbers
df = df.apply(pd.to_numeric)

# Drop rows that still have missing values
df = df.dropna()

# Scatterplot: Weight vs MPG (EDA)
plt.figure()
plt.scatter(df["weight"], df["mpg"])
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("Weight vs MPG")
plt.show()

# Choose features and target
X = df.drop(columns=["mpg"])
y = df["mpg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the mean squared error
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot Actual vs Predicted MPG
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")

# Use only weight for a simple regression line visualization
X_weight = df[["weight"]]

# Train simple model on weight only (for visualization)
model_vis = LinearRegression()
model_vis.fit(X_weight, y)

# Create a smooth line of predicted MPG values
weight_range = pd.DataFrame(
    {"weight": np.linspace(df["weight"].min(), df["weight"].max(), 100)}
)
mpg_pred_line = model_vis.predict(weight_range)

# Plot Weight vs MPG with a fitted regression line
plt.figure()
plt.scatter(X_weight, y, alpha=0.6, label="Actual Data")
plt.plot(weight_range, mpg_pred_line, color="red", linewidth=2, label="Predicted Trend")

plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("Weight vs MPG with Regression Model")
plt.legend()
plt.show()
