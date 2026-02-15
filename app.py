import streamlit as st
import pandas as pd
import joblib
import os
import pickle

# -----------------------------------
# App Title
# -----------------------------------
st.title("BITS ML Assignment 2 - Classification Models")
st.write("Comparison of 6 Machine Learning Models")

# -----------------------------------
# Models folder path
# -----------------------------------
MODELS_FOLDER = "models"
results_path = os.path.join(MODELS_FOLDER, "results.csv")
feature_path = os.path.join(MODELS_FOLDER, "feature_names.pkl")

# -----------------------------------
# Check if results.csv exists
# -----------------------------------
if not os.path.exists(results_path):
    st.error("results.csv not found in models folder.")
    st.stop()

# Load results
results = pd.read_csv(results_path)

# Display results table
st.subheader("Model Performance Metrics")
st.dataframe(results)

# -----------------------------------
# Model selection
# -----------------------------------
model_name = st.selectbox(
    "Select Model for Prediction",
    results["Model"]
)

model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")

if not os.path.exists(model_path):
    st.error(f"{model_name}.pkl not found in models folder.")
    st.stop()

# Load selected model
model = joblib.load(model_path)

# -----------------------------------
# Load feature names
# -----------------------------------
if not os.path.exists(feature_path):
    st.error("feature_names.pkl not found. Please retrain and save features.")
    st.stop()

with open(feature_path, "rb") as f:
    feature_names = pickle.load(f)

# -----------------------------------
# Sample CSV Download Section
# -----------------------------------
st.subheader("Download Sample CSV Format")

sample_df = pd.DataFrame(columns=feature_names)

st.download_button(
    label="Download Sample Input CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_input.csv",
    mime="text/csv"
)

# -----------------------------------
# File upload section
# -----------------------------------
st.subheader("Upload CSV File for Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file with same feature columns",
    type=["csv"]
)

# -----------------------------------
# Prediction logic
# -----------------------------------
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        # Check empty file
        if data.shape[0] == 0:
            st.error("Uploaded CSV contains no data rows. Please upload a valid file.")
            st.stop()

        # Ensure correct columns
        missing_cols = set(feature_names) - set(data.columns)

        if missing_cols:
            st.error("Missing required columns:")
            st.write(missing_cols)
            st.stop()

        # Keep correct column order
        data = data[feature_names]

        # Make predictions
        predictions = model.predict(data)

        # Show success message
        st.success("Prediction Successful!")

        # Display results
        result_df = data.copy()
        result_df["Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        # Download predictions
        st.download_button(
            label="Download Predictions",
            data=result_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.write("Error details:", str(e))

# -----------------------------------
# Footer
# -----------------------------------
st.write("---")
st.write("Developed for BITS Machine Learning Assignment 2")
