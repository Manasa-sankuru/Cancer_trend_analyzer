import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(r"E:\country\death .csv")

# ------------------------
# Define target and features
# ------------------------
target_col = 'Recent Trend (2)'
y = df[target_col]
X = df.drop([target_col], axis=1)

# ------------------------
# Train-test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Convert all numeric columns
# ------------------------
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

# ------------------------
# Encode target
# ------------------------
encoder = TargetEncoder()
y_train_encoded = encoder.fit_transform(y_train, y_train)
y_test_encoded = encoder.transform(y_test, y_test)

# ------------------------
# Scale features
# ------------------------
scale_features = [
    'index', 'County', 'FIPS', 'Met Objective of 45.5? (1)',
    'Age-Adjusted Death Rate', 'Lower 95% Confidence Interval for Death Rate',
    'Upper 95% Confidence Interval for Death Rate', 'Average Deaths per Year',
    'Recent 5-Year Trend (2) in Death Rates',
    'Lower 95% Confidence Interval for Trend', 'Upper 95% Confidence Interval for Trend'
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_features] = scaler.fit_transform(X_train[scale_features])
X_test_scaled[scale_features] = scaler.transform(X_test[scale_features])

# ------------------------
# Train Random Forest
# ------------------------
rf_params = {
    'n_estimators': [100, 250],
    'max_depth': [10, 20],
    'criterion': ["squared_error"]
}

rf_cv = GridSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42), rf_params, cv=3)
rf_cv.fit(X_train_scaled, y_train_encoded)

rf_model = RandomForestRegressor(**rf_cv.best_params_, n_jobs=-1, random_state=42)
rf_model.fit(X_train_scaled, y_train_encoded)

# ------------------------
# Evaluate
# ------------------------
rf_pred = rf_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test_encoded, rf_pred))
print(f"Random Forest RMSE: {rmse:.4f}")

# ------------------------
# Save model & preprocessor
# ------------------------
joblib.dump(rf_model, r"E:\country\cancer_model.pkl")
joblib.dump(scaler, r"E:\country\scaler.pkl")
joblib.dump(encoder, r"E:\country\target_encoder.pkl")

print("Model, scaler, and target encoder saved successfully.")
