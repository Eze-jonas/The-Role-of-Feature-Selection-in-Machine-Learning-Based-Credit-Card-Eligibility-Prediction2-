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

# Absolute path to your saved columns folder
saved_columns_path = os.getenv("SAVED_COLUMNS_PATH", "saved_columns")

# Load transformation objects with the absolute path
education_encoder = joblib.load(os.path.join(saved_columns_path, 'label_encoding_Education_Type.pkl'))
education_scaler = joblib.load(os.path.join(saved_columns_path, 'scaler_Education_Type.pkl'))
tbd_transformer = joblib.load(os.path.join(saved_columns_path, 'transformer_Total_Bad_Debt.pkl'))
tbd_scaler = joblib.load(os.path.join(saved_columns_path, 'robust_scaler_Total_Bad_Debt.pkl'))
tgd_transformer = joblib.load(os.path.join(saved_columns_path, 'transformer_Total_Good_Debt.pkl'))

# Load GA-selected features with the absolute path
ga_selected_features_path = os.path.join(saved_columns_path, 'ga_selected_features.pkl')
if os.path.exists(ga_selected_features_path):
    ga_selected_features = joblib.load(ga_selected_features_path).get("selected_features", [])
else:
    print("Warning: GA-selected features file not found. Using default features.")
    ga_selected_features = [
        "Owned_Realty", "Education_Type", "Owned_Mobile_Phone", "Total_Bad_Debt", "Total_Good_Debt",
        "Applicant_Gender_M", "Income_Type_Pensioner", "Income_Type_Student", "Income_Type_Working",
        "Family_Status_Married", "Family_Status_Separated", "Family_Status_Single / not married",
        "Housing_Type_Municipal apartment", "Housing_Type_With parents", "Job_Title_Cleaning staff",
        "Job_Title_Core staff", "Job_Title_High skill tech staff", "Job_Title_IT staff", "Job_Title_Laborers",
        "Job_Title_Managers", "Job_Title_Medicine staff", "Job_Title_Realty agents", "Job_Title_Secretaries",
        "Job_Title_Security staff"
    ]

# Expected Features (from training)
expected_features = [
    "Owned_Realty", "Education_Type", "Owned_Mobile_Phone", "Total_Bad_Debt", "Total_Good_Debt",
    "Applicant_Gender_M", "Income_Type_Pensioner", "Income_Type_Student", "Income_Type_Working",
    "Family_Status_Married", "Family_Status_Separated", "Family_Status_Single / not married",
    "Housing_Type_Municipal apartment", "Housing_Type_With parents", "Job_Title_Cleaning staff",
    "Job_Title_Core staff", "Job_Title_High skill tech staff", "Job_Title_IT staff", "Job_Title_Laborers",
    "Job_Title_Managers", "Job_Title_Medicine staff", "Job_Title_Realty agents", "Job_Title_Secretaries",
    "Job_Title_Security staff"
]

binary_columns = [
    "Owned_Realty", "Owned_Mobile_Phone", "Applicant_Gender_M", "Income_Type_Pensioner",
    "Income_Type_Student", "Income_Type_Working", "Family_Status_Married", "Family_Status_Separated",
    "Family_Status_Single / not married", "Housing_Type_Municipal apartment", "Housing_Type_With parents",
    "Job_Title_Cleaning staff", "Job_Title_Core staff", "Job_Title_High skill tech staff", "Job_Title_IT staff",
    "Job_Title_Laborers", "Job_Title_Managers", "Job_Title_Medicine staff", "Job_Title_Realty agents",
    "Job_Title_Secretaries", "Job_Title_Security staff"
]

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "fallback_secret")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=6)
jwt = JWTManager(app)

# Dummy users (replace with database)
users = {"admin": "password123"}

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

        # Apply label encoding and standardization to Education_Type
        if "Education_Type" in data:
            education_value = data["Education_Type"]
            if education_value in education_encoder['encoding_mapping']:
                encoded_value = education_encoder['encoding_mapping'][education_value]
                data["Education_Type"] = education_scaler.transform([[encoded_value]])[0][0]
            else:
                return jsonify({"error": "Invalid Education_Type value"}), 400

        # Apply Winsorization, Yeo-Johnson transformation, and robust scaling to Total_Bad_Debt
        if "Total_Bad_Debt" in data:
            tbd_value = min(float(data["Total_Bad_Debt"]), 5)  # Clipping at 5
            transformed_value = tbd_transformer.transform([[tbd_value]])[0][0]
            data["Total_Bad_Debt"] = tbd_scaler.transform([[transformed_value]])[0][0]

        # Apply Yeo-Johnson transformation to Total_Good_Debt
        if "Total_Good_Debt" in data:
            tgd_value = float(data["Total_Good_Debt"])
            data["Total_Good_Debt"] = tgd_transformer.transform([[tgd_value]])[0][0]

        # Convert to DataFrame and ensure correct column order
        df = pd.DataFrame([data]).reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]
        df["Status"] = "Eligible" if prediction == 1 else "Not Eligible"

        # Convert binary values back to Yes/No
        df[binary_columns] = df[binary_columns].replace({1: "Yes", 0: "No"})

        return jsonify(df.iloc[0].to_dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 400

 #Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=False)