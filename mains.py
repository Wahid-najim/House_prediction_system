import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Step 1: Load a real estate dataset
df = pd.read_csv('datasets.csv')

# Display the first few rows of the dataset
print(df.head())

# Step 2: Split data into training and testing sets
X = df.drop('medv', axis=1)
y = df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define preprocessing pipelines for numerical data
num_cols = X.columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing numerical data with median
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Step 4: Define the Random Forest model and pipeline
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest regressor model

model_pipeline = Pipeline([
    ('preprocessor', num_pipeline),
    ('regressor', rf)
])

# Step 5: Train the model
model_pipeline.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model_pipeline.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared score

print('Random Forest Model Evaluation:')
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R²):', r2)

# Step 8: Visualization with seaborn

# 1. Actual vs. Predicted Values Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Median House Price ($1000s)')
plt.ylabel('Predicted Median House Price ($1000s)')
plt.show()

# 2. Residuals Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=y_test - y_pred)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted Median House Price ($1000s)')
plt.ylabel('Residuals ($1000s)')
plt.show()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True)
plt.title('Error Distribution')
plt.xlabel('Residuals ($1000s)')
plt.ylabel('Frequency')
plt.show()
