from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load(
    "model/insurance_model.pkl"
)


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Medical Insurance Expense Prediction API",
    description="Linear Regression based medical insurance expense prediction",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class InsuranceInput(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=100
    )

    gender: str

    bmi: float = Field(
        ...,
        gt=0
    )

    children: int = Field(
        ...,
        ge=0,
        le=20
    )

    discount_eligibility: str

    region: str


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Medical Insurance Prediction API",
        "status": "running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: InsuranceInput):

    input_data = pd.DataFrame([
        {
            "age": data.age,
            "gender": data.gender,
            "bmi": data.bmi,
            "children": data.children,
            "discount_eligibility":
                data.discount_eligibility,
            "region": data.region
        }
    ])

    prediction = model.predict(
        input_data
    )[0]

    return {
        "predicted_expenses":
            round(float(prediction), 2)
    }