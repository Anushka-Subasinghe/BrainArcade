import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('dataset.csv')

# Adjust the category labels to start from 0 instead of 1
df['category'] = df['category'] - 1

# Separate features and target variable
X = df.drop('category', axis=1)  # Features
y = df['category']  # Target variable (category)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost classifier
xgb_classifier = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',  # Ensure you're using an evaluation metric
    n_jobs=-1  # Use all available cores for parallel processing
)

# Define a smaller hyperparameter grid to reduce training time
param_grid = {
    'n_estimators': [100, 150],  # Reduced range
    'max_depth': [3, 5],  # Reduced range
    'learning_rate': [0.1, 0.2],  # Reduced range
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_xgb = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predict on the test set
y_pred = best_xgb.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report to see precision, recall, F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
import joblib
joblib.dump(best_xgb, 'trained_model.joblib')
