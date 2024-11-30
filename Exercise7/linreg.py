import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Preprocess the data
def preprocess_data(df):
    df = df.dropna()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce').astype(int) / 10**9
    numeric_cols = ['soil_mois', 'temp', 'soil_humid', 'air_temp', 'wind_speed', 'air_humid', 
                    'wind_gust', 'pressure', 'ph', 'rainfall', 'n', 'p', 'k']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)
    return df

# Load data
df = pd.read_csv('trap.csv')

# Preprocess data
df = preprocess_data(df)

# Define models
models = [
    (['temp'], 'soil_mois'),
    (['temp', 'air_temp'], 'soil_mois'),
    (['temp', 'air_temp', 'wind_speed'], 'soil_mois'),
    (['soil_humid', 'air_temp', 'wind_speed'], 'soil_mois'),
    (['ph', 'rainfall', 'n', 'p', 'k'], 'soil_mois')
]

# Train and evaluate models
for i, (features, target) in enumerate(models, 1):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model {i} - Features: {features}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    plt.title(f'Model {i} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()
