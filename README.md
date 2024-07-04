

---

# Housing Price Prediction

## Introduction
This project involves building a linear regression model to predict housing prices based on features such as area, number of bedrooms, and number of bathrooms. The dataset used for this project is stored in a CSV file.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/housing-price-prediction.git
   cd housing-price-prediction
   ```
2. **Install the required libraries:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. **Ensure the dataset is available:**
   - The dataset should be stored in a CSV file named `Housing.csv` in the specified path (`C:\\Users\\Gaura\\Documents\\COLLEGE\\PYTHON WORKSHOP\\Housing.csv`).

## Usage
1. **Run the script:**
   ```bash
   python housing_price_prediction.py
   ```
2. **Script Explanation:**

   - **Loading the dataset:**
     ```python
     data = pd.read_csv("C:\\Users\\Gaura\\Documents\\COLLEGE\\PYTHON WORKSHOP\\Housing.csv", on_bad_lines='skip')
     ```

   - **Extracting features and target variable:**
     ```python
     X = data[['area', 'bedrooms', 'bathrooms']]
     y = data['price']
     ```

   - **Splitting the dataset into training and testing sets:**
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

   - **Creating and fitting the linear regression model:**
     ```python
     model = LinearRegression()
     model.fit(X_train, y_train)
     ```

   - **Predicting prices using the model:**
     ```python
     y_pred = model.predict(X_test)
     ```

   - **Calculating the coefficients and intercept:**
     ```python
     coefficients = model.coef_
     intercept = model.intercept_
     print("Model Equation:")
     print(f"price = {coefficients[0]:.2f} * area + {coefficients[1]:.2f} * bedrooms + {coefficients[2]:.2f} * bathrooms + {intercept:.2f}")
     ```

   - **Calculating the Mean Squared Error and R-squared value:**
     ```python
     mse = mean_squared_error(y_test, y_pred)
     r2 = r2_score(y_test, y_pred)
     print(f"\nMean Squared Error: {mse:.2f}")
     print(f"R-squared: {r2:.2f}")
     ```

   - **Visualizing the data and the regression line:**
     ```python
     fig, ax = plt.subplots(1, 2, figsize=(12, 6))

     ax[0].scatter(X_test['area'], y_test, color='blue', label='Actual')
     ax[0].set_xlabel('area')
     ax[0].set_ylabel('price')
     ax[0].set_title('Actual vs. Predicted Prices')
     ax[0].legend()

     ax[1].scatter(X_test['area'], y_pred, color='red', label='Predicted')
     ax[1].set_xlabel('area')
     ax[1].set_ylabel('price')
     ax[1].set_title('Actual vs. Predicted Prices')
     ax[1].legend()

     x_range = np.linspace(X_test['area'].min(), X_test['area'].max(), 100)
     y_range = coefficients[0] * x_range + intercept
     ax[1].plot(x_range, y_range, color='green', label='Regression Line')

     plt.show()
     ```

## Output
- The script prints the model equation, mean squared error, and R-squared value.
- It also visualizes the actual and predicted housing prices, and plots the regression line.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify this README file to better suit your project's specific needs.
