import streamlit as st
import numpy as np
import joblib

pipeline = joblib.load("poly_linear_regression_pipeline.pkl")

st.title("IPL Player Runs Prediction App (Polynomial Regression)")

inns = st.number_input("Innings", min_value=0)
no = st.number_input("Not Outs", min_value=0)
avg = st.number_input("Batting Average", min_value=0.0)
bf = st.number_input("Balls Faced", min_value=0)
fours = st.number_input("Number of 4s", min_value=0)
sixes = st.number_input("Number of 6s", min_value=0)

if st.button("Predict Runs"):
    input_data = np.array([[inns, no, avg, bf, fours, sixes]])
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted Runs: {prediction[0]:.2f}")
