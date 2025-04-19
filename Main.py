from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

uploaded_csv = None  
sample_data = None  
important_features = []

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_csv, sample_data, important_features

    if request.method == "POST":
        file = request.files["file"]
        if file:
            uploaded_csv = pd.read_csv(file)

            # Drop unnecessary columns
            cleaned_df = uploaded_csv.drop(columns=["Transaction_ID", "Merchant_ID", "Customer_ID", "Device_ID", "IP_Address", "Date", "Time"], errors="ignore")

            if "fraud" not in cleaned_df.columns:
                return render_template("index.html", message="Error: CSV must contain 'fraud' column.")

            # Encode categorical columns
            categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
            label_encoders = {}
            for col in categorical_cols:
                encoder = LabelEncoder()
                cleaned_df[col] = encoder.fit_transform(cleaned_df[col].astype(str))
                label_encoders[col] = encoder

            # Separate features and target
            X = cleaned_df.drop(columns=["fraud"])
            y = cleaned_df["fraud"]

            # Scale numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train new model
            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
            model.fit(X_scaled, y)

            # Save model & preprocessing objects
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(label_encoders, "label_encoders.pkl")
            joblib.dump(X.columns.tolist(), "feature_names.pkl")

            # Get most important features (Top 5)
            feature_importance = model.feature_importances_
            important_features = list(X.columns[feature_importance.argsort()[-5:]])  # Select Top 5 features

            # Show sample data
            sample_data = uploaded_csv[important_features].head(5).to_dict(orient="records")

            return render_template("index.html", message="Model trained successfully!", csv_data=sample_data, important_features=important_features)

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists("model.pkl"):
        return render_template("index.html", message="Error: Train model first by uploading a CSV!")

    # Load fresh model & preprocessing objects
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")

    # Get manual input
    input_data = {key: request.form[key] for key in request.form}

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df:
            df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    # Ensure correct columns
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numerical features
    df_scaled = scaler.transform(df)

    # Predict fraud
    prediction = model.predict(df_scaled)[0]

    return render_template("index.html", prediction_text="🚨 Fraudulent" if prediction == 1 else "✅ Legitimate", csv_data=sample_data, important_features=important_features)

if __name__ == "__main__":
    app.run(debug=True)
