# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\Acer25\OneDrive\Desktop\RAJ\codsoft_projects\SALES PREDICTION USING PYTHON-4\advertising.csv")  # Update the path if needed

# Display the first few rows
print("Dataset preview:\n", data.head())

# Step 2: Exploratory Data Analysis
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Visualizing the relationship between TV advertising and Sales
sns.scatterplot(data=data, x="TV", y="Sales")
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Cost (in thousands)")
plt.ylabel("Sales (in thousands)")
plt.show()

# Step 3: Data Preparation
X = data[["TV"]]  # Feature
y = data["Sales"]  # Target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building and Trainingm 
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Making Predictions
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Step 7: Visualizing Predictions
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.title("TV Advertising vs Sales (Test Set)")
plt.xlabel("TV Advertising Cost (in thousands)")
plt.ylabel("Sales (in thousands)")
plt.legend()
plt.show()
