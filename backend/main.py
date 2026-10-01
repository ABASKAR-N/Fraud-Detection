from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="AI-based Credit Card Fraud Detection using Random Forest",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model = joblib.load("model.pkl")

    print("Random Forest model loaded successfully!")

except Exception as e:

    model = None

    print("Error loading model:")
    print(e)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float

    V6: float
    V7: float
    V8: float
    V9: float
    V10: float

    Amount: float


# ============================================================
# HOME API
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "model": "Random Forest"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is not None:

        return {
            "status": "healthy",
            "model_loaded": True,
            "model": "Random Forest"
        }

    else:

        return {
            "status": "error",
            "model_loaded": False
        }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    if model is None:

        return {
            "status": "error",
            "message": "Model is not loaded"
        }


    # --------------------------------------------------------
    # Convert input into DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame([{

        "Time": transaction.Time,

        "V1": transaction.V1,
        "V2": transaction.V2,
        "V3": transaction.V3,
        "V4": transaction.V4,
        "V5": transaction.V5,

        "V6": transaction.V6,
        "V7": transaction.V7,
        "V8": transaction.V8,
        "V9": transaction.V9,
        "V10": transaction.V10,

        "Amount": transaction.Amount

    }])


    # --------------------------------------------------------
    # Random Forest Prediction
    # --------------------------------------------------------

    prediction = model.predict(data)

    probability = model.predict_proba(data)


    # Probability of fraud
    fraud_probability = probability[0][1] * 100


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction[0] == 1:

        result = "FRAUDULENT"

    else:

        result = "LEGITIMATE"


    # --------------------------------------------------------
    # Return JSON
    # --------------------------------------------------------

    return {

        "status": "success",

        "prediction": int(prediction[0]),

        "result": result,

        "fraud_probability": round(
            float(fraud_probability),
            2
        )

    }