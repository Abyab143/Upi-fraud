import joblib
import numpy as np
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# Function to preprocess and predict transactions
def predict_transaction(input_data):
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
    transaction_ids = df["Transaction_ID"] if "Transaction_ID" in df else None
    
    # Drop Transaction_ID for model input
    df = df.drop(columns=["Transaction_ID"], errors='ignore')
    
    # Encode categorical variables with handling for unseen categories
    for col, encoder in label_encoders.items():
        if col in df:
            df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
    # Ensure the input data matches the features used during training
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale numerical features
    df_scaled = scaler.transform(df)
    
    # Predict fraud
    predictions = model.predict(df_scaled)
    
    result = ["\U0001F6A8 Fraudulent" if pred == 1 else "✅ Legitimate" for pred in predictions]
    if transaction_ids is not None:
        return list(zip(transaction_ids, result))
    return result

# Manual input example
input_data = {
    "Transaction_ID": "TXN123456",
    "Transaction_Type": "Bank Transfer",
    "Payment_Gateway": "UPI Pay",
    "Transaction_City": "Mumbai",
    "Transaction_State": "Maharashtra",
    "Device_OS": "Android",
    "Transaction_Frequency": 5,
    "Merchant_Category": "Retail",
    "Transaction_Channel": "Online",
    "Transaction_Amount_Deviation": 10.5,
    "Days_Since_Last_Transaction": 3,
    "amount": 500
}

# Predict manually entered transaction
print("Transaction Status:", predict_transaction(input_data))

# CSV file input example
csv_file_path = "Dataset/data.csv"  # Update with the correct file path
df_csv = pd.read_csv(csv_file_path)
predictions_csv = predict_transaction(df_csv)

df_result = pd.DataFrame(predictions_csv, columns=["Transaction_ID", "Fraud_Prediction"])
df_result.to_csv("Dataset/output.csv", index=False)

print("Predictions saved to output.csv")
