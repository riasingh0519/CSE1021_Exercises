import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Preprocess data
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

# Convert target to category
df['status'] = df['status'].astype('category')

# Define models
models = [
    (['temp'], 'status'),
    (['temp', 'air_temp'], 'status'),
    (['temp', 'air_temp', 'wind_speed'], 'status'),
    (['soil_humid', 'air_temp', 'wind_speed'], 'status'),
    (['ph', 'rainfall', 'n', 'p', 'k'], 'status')
]

# Train and evaluate models
for i, (features, target) in enumerate(models, 1):
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # KNN model
    model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model {i} - Features: {features}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    print()

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Model {i} - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, np.unique(y), rotation=45)
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
