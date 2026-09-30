import streamlit as st
import requests


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏥 Medical Insurance Expense Predictor")

st.write(
    "Enter customer information to predict "
    "estimated medical insurance expenses."
)


# --------------------------------------------------
# Input fields
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

gender = st.selectbox(
    "Gender",
    [
        "male",
        "female"
    ]
)

bmi = st.number_input(
    "BMI",
    min_value=1.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

discount_eligibility = st.selectbox(
    "Discount Eligibility",
    [
        "yes",
        "no"
    ]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)


# --------------------------------------------------
# API URL
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

if st.button(
    "Predict Medical Expenses",
    use_container_width=True
):

    payload = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "discount_eligibility":
            discount_eligibility,
        "region": region
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            predicted_expenses = result[
                "predicted_expenses"
            ]

            st.success(
                "Prediction completed successfully!"
            )

            st.metric(
                "Estimated Medical Expenses",
                f"{predicted_expenses:,.2f}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure the API is running."
        )

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )