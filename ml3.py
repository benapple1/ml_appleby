from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
dataset = load_dataset("scikit-learn/auto-mpg")
df = dataset["train"].to_pandas()

# Inspect columns
print(df.columns)

# Clean data
df = df.drop(columns=["name"], errors="ignore")
df = df.dropna()

# 🔹 Scatterplot: Weight vs MPG (EDA)
plt.figure()
plt.scatter(df["weight"], df["mpg"])
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("Weight vs MPG")
plt.show()

# Features & target
X = df.drop(columns=["mpg"])
y = df["mpg"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))

# Predicted vs Actual plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()
