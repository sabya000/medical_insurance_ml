import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/medical_insurance.csv")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Data cleaning
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate records if appropriate for your dataset
df = df.drop_duplicates().copy()


# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

# We are predicting expenses.
# Premium is NOT used as an input because it is
# derived from expenses and would cause target leakage.

X = df[
    [
        "age",
        "gender",
        "bmi",
        "children",
        "discount_eligibility",
        "region"
    ]
]

y = df["expenses"]


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

numeric_features = [
    "age",
    "bmi",
    "children"
]

categorical_features = [
    "gender",
    "discount_eligibility",
    "region"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 6. Linear Regression model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel Evaluation")
print("----------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))


# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

joblib.dump(
    model,
    "model/insurance_model.pkl"
)

print("\nModel saved successfully!")
print("Location: model/insurance_model.pkl")