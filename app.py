from flask import Flask, request, jsonify, render_template
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model file paths
MODEL_PATH = os.getenv("DECISION_TREE_MODEL_PATH", "best_decision_tree_model.joblib")

# Load the model with error handling
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "fallback_secret")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=6)
jwt = JWTManager(app)

# Dummy users (replace with database)
users = {"admin": "password123"}

# Expected Features (from training)
expected_features = [
    "Owned_Car", "Owned_Email", "Applicant_Age", "Years_of_Working",
    "Total_Bad_Debt", "Total_Good_Debt", "Income_Type_Pensioner", "Income_Type_State servant",
    "Family_Status_Married", "Housing_Type_Office apartment", "Housing_Type_Rented apartment",
    "Job_Title_Cooking staff", "Job_Title_Core staff", "Job_Title_High skill tech staff",
    "Job_Title_Laborers", "Job_Title_Low-skill Laborers", "Job_Title_Medicine staff",
    "Job_Title_Private service staff", "Job_Title_Sales staff", "Job_Title_Secretaries",
    "Job_Title_Waiters/barmen staff"
]

# Binary columns to be converted
binary_columns = [
    "Owned_Car", "Owned_Email", "Income_Type_Pensioner", "Income_Type_State servant",
    "Family_Status_Married", "Housing_Type_Office apartment", "Housing_Type_Rented apartment",
    "Job_Title_Cooking staff", "Job_Title_Core staff", "Job_Title_High skill tech staff",
    "Job_Title_Laborers", "Job_Title_Low-skill Laborers", "Job_Title_Medicine staff",
    "Job_Title_Private service staff", "Job_Title_Sales staff", "Job_Title_Secretaries",
    "Job_Title_Waiters/barmen staff"
]

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for login & token generation
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if username in users and users[username] == password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)

    return jsonify({"message": "Invalid credentials"}), 401

# Prediction route (protected)
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check logs."}), 500

    try:
        data = request.json

        # Convert Yes/No values in binary columns
        for col in binary_columns:
            if col in data:
                data[col] = 1.0 if str(data[col]).strip().lower() == "yes" else 0.0

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure all expected features exist
        missing_features = list(set(expected_features) - set(df.columns))
        for col in missing_features:
            df[col] = 0  # Default missing features to 0

        # Ensure correct column order
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)[0]
        df["Status"] = "Eligible" if prediction == 1 else "Not Eligible"

        # Convert binary values back to Yes/No
        df[binary_columns] = df[binary_columns].replace({1: "Yes", 0: "No"})

        return jsonify(df.iloc[0].to_dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
#if __name__ == "__main__":
    #app.run(debug=True)

if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), deburg = False)