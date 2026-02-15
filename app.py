import streamlit as st
import pandas as pd
import joblib
import os

st.title("BITS ML Assignment 2 - Classification Models")

st.write("Comparison of 6 Machine Learning Models")

# path to models folder
MODELS_FOLDER = "models"

# check if results.csv exists
results_path = os.path.join(MODELS_FOLDER, "results.csv")

if os.path.exists(results_path):

    results = pd.read_csv(results_path)

    st.subheader("Model Performance Metrics")

    st.dataframe(results)

    # dropdown
    model_name = st.selectbox(
        "Select Model",
        results["Model"]
    )

    # load selected model
    model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        st.subheader("Upload CSV file for prediction")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:

            data = pd.read_csv(uploaded_file)

            predictions = model.predict(data)

            st.write("Predictions:")
            st.write(predictions)

    else:

        st.error("Model file not found")

else:

    st.error("results.csv not found in models folder")
