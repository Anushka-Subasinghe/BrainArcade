from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

# Separate features and target variable
X = df.drop('category', axis=1)
y = df['category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for balancing the classes
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# Define a dictionary to store models and their performance
models = {}

# Define and tune Random Forest model (included from previous code)
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train_res, y_train_res)
best_rf = rf_grid_search.best_estimator_
models['Random Forest'] = (best_rf, None)  # (model, accuracy)

# Define and tune Logistic Regression
lr_clf = LogisticRegression(random_state=42)
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid_search = GridSearchCV(lr_clf, lr_param_grid, cv=5, n_jobs=-1)
lr_grid_search.fit(X_train_res, y_train_res)
best_lr = lr_grid_search.best_estimator_
models['Logistic Regression'] = (best_lr, None)  # (model, accuracy)

# Define and tune Gradient Boosting (included from previous code)
# ... (similar code for Gradient Boosting)

# Define and tune Support Vector Machine
svc_clf = SVC(random_state=42)
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1]
}
svc_grid_search = GridSearchCV(svc_clf, svc_param_grid, cv=5, n_jobs=-1)
svc_grid_search.fit(X_train_res, y_train_res)
best_svc = svc_grid_search.best_estimator_
models['SVM'] = (best_svc, None)  # (model, accuracy)

# Define and tune K-Nearest Neighbors
knn_clf = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': range(1, 21),
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}
knn_grid_search = GridSearchCV(knn_clf, knn_param_grid, cv=5, n_jobs=-1)
knn_grid_search.fit(X_train_res, y_train_res)
best_knn = knn_grid_search.best_estimator_
models['K-Nearest Neighbors'] = (best_knn, None)  # (model, accuracy)

# Evaluate each model on the test set and update the dictionary
for model_name, (model, _) in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    models[model_name] = (model, accuracy)

# Find the best model based on accuracy
best_model_name = max(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]

# Print the results
print("Best Model:", best_model_name)
print("Accuracy:", models[best_model_name][1])
print("\nClassification Report:\n", classification_report(y_test, best_model.predict(X_test_scaled)))

# Save the best model and scaler
joblib.dump(best_model, 'trained_model.pkl')