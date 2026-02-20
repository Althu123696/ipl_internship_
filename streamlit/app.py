import streamlit as st
import requests

st.title("IPL Player Runs Prediction")

# Input fields
Inns = st.number_input("Innings", min_value=0)
NO = st.number_input("Not Outs", min_value=0)
Avg = st.number_input("Average", min_value=0.0)
BF = st.number_input("Balls Faced", min_value=0)
Fours = st.number_input("Number of 4s", min_value=0)
Sixes = st.number_input("Number of 6s", min_value=0)

# Use FastAPI service name when running inside Docker Compose
FASTAPI_URL = "http://fastapi:8000/predict"

if st.button("Predict Runs"):
    try:
        response = requests.post(
            FASTAPI_URL,
            json={
                "Inns": Inns,
                "NO": NO,
                "Avg": Avg,
                "BF": BF,
                "4s": Fours,
                "6s": Sixes
            }
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Runs: {result['Predicted Runs']:.2f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Is it running in Docker Compose on port 8000?")
