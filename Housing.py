import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the CSV file
data = pd.read_csv("C:\\Users\Gaura\Documents\COLLEGE\PYTHON WORKSHOP\Housing.csv", on_bad_lines='skip')

# Extract the features and target variable
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices using the model
y_pred = model.predict(X_test)

# Calculate the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the model equation
print("Model Equation:")
print(f"price = {coefficients[0]:.2f} * area + {coefficients[1]:.2f} * bedrooms + {coefficients[2]:.2f} * bathrooms + {intercept:.2f}")

# Calculate the Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualization of the data and the regression line
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of the actual data
ax[0].scatter(X_test['area'], y_test, color='blue', label='Actual')
ax[0].set_xlabel('area')
ax[0].set_ylabel('price')
ax[0].set_title('Actual vs. Predicted Prices')
ax[0].legend()

# Scatter plot of the predicted data
ax[1].scatter(X_test['area'], y_pred, color='red', label='Predicted')
ax[1].set_xlabel('area')
ax[1].set_ylabel('price')
ax[1].set_title('Actual vs. Predicted Prices')
ax[1].legend()

# Plot the regression line
x_range = np.linspace(X_test['area'].min(), X_test['area'].max(), 100)
y_range = coefficients[0] * x_range + intercept
ax[1].plot(x_range, y_range, color='green', label='Regression Line')

plt.show()
