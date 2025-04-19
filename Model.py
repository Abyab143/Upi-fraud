import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Dataset/data.csv")

# Preserve Transaction_ID for reference
transaction_ids = df["Transaction_ID"]

# Drop unnecessary columns
df_cleaned = df.drop(columns=["Transaction_ID", "Merchant_ID", "Customer_ID", "Device_ID", "IP_Address", "Date", "Time"])

# Handle missing values (fill with median for numerical, "Unknown" for categorical)
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col].fillna("Unknown", inplace=True)
    else:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Convert categorical variables to numerical using Label Encoding
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].astype(str)  # Ensure values are strings
    encoder = LabelEncoder()
    df_cleaned[col] = encoder.fit_transform(df_cleaned[col])
    label_encoders[col] = encoder.classes_  # Save unique classes

# Separate features and target variable
X = df_cleaned.drop(columns=["fraud"])
y = df_cleaned["fraud"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize RandomForest parameters using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Use the best model found
model = grid_search.best_estimator_

# Save model and preprocessing objects as a single file
model_data = {
    "model": model,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_names": X.columns.tolist()
}

joblib.dump(model_data, "fraud_detection_model.pkl")

print("✅ Model training complete. Saved as 'fraud_detection_model.pkl'.")

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Print confusion matrix and classification report
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Analysis
importances = model.feature_importances_
feature_names = X.columns

# Sort and plot feature importances
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance in Fraud Detection")
plt.show()
