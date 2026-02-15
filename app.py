import streamlit as st
import pandas as pd
import joblib
import os
import pickle

st.title("BITS ML Assignment 2 - Classification Models")
st.write("Comparison of 6 Machine Learning Models")

MODELS_FOLDER = "models"

results_path = os.path.join(MODELS_FOLDER, "results.csv")

if os.path.exists(results_path):

    results = pd.read_csv(results_path)

    st.subheader("Model Performance Metrics")
    st.dataframe(results)

    model_name = st.selectbox(
        "Select Model",
        results["Model"]
    )

    model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        # ✅ Load feature names
        feature_path = os.path.join(MODELS_FOLDER, "feature_names.pkl")

        if os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                feature_names = pickle.load(f)
        else:
            st.error("feature_names.pkl not found. Please retrain and save features.")
            st.stop()

        st.subheader("Upload CSV file for prediction")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:

            try:
                data = pd.read_csv(uploaded_file)

                # ✅ Keep only required columns in correct order
                data = data[feature_names]

                predictions = model.predict(data)

                st.write("Predictions:")
                st.write(predictions)

            except Exception as e:
                st.error("Uploaded CSV format does not match training data.")
                st.write("Error details:", str(e))

    else:
        st.error("Model file not found")

else:
    st.error("results.csv not found in models folder")
