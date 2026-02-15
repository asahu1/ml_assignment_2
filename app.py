import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BITS ML Assignment 2",
    page_icon="🤖",
    layout="wide"
)

st.title("BITS ML Assignment 2 - Classification Models")

st.write(
    "This application compares multiple Machine Learning classification models "
    "and allows predictions on uploaded test datasets."
)

# =========================
# DOWNLOAD SAMPLE FILE
# =========================

st.subheader("Download Sample Test File")

try:
    with open("test_prediction.csv", "rb") as file:
        st.download_button(
            label="Download test_prediction.csv",
            data=file,
            file_name="test_prediction.csv",
            mime="text/csv"
        )
except:
    st.warning("test_prediction.csv not found in repository.")

# =========================
# LOAD RESULTS
# =========================

MODELS_FOLDER = "models"
results_path = os.path.join(MODELS_FOLDER, "results.csv")

if not os.path.exists(results_path):

    st.error("results.csv not found in models folder")
    st.stop()

results = pd.read_csv(results_path)

st.subheader("Model Performance Metrics")

st.dataframe(results, use_container_width=True)

# =========================
# MODEL SELECTION
# =========================

model_name = st.selectbox(
    "Select Model for Prediction",
    results["Model"]
)

model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")

if not os.path.exists(model_path):

    st.error(f"{model_name}.pkl not found.")
    st.stop()

model = joblib.load(model_path)

# =========================
# FILE UPLOAD
# =========================

st.subheader("Upload CSV File for Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# =========================
# PREDICTION LOGIC
# =========================

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        if data.empty:

            st.error("Uploaded CSV contains no data rows.")
            st.stop()

        st.subheader("Uploaded Data Preview")

        st.dataframe(data.head(), use_container_width=True)

        predictions = model.predict(data)

        result_df = data.copy()
        result_df["Prediction"] = predictions

        st.subheader("Prediction Results")

        st.dataframe(result_df, use_container_width=True)

        # download results

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success("Prediction completed successfully.")

    except Exception as e:

        st.error("Prediction failed.")
        st.error(str(e))

# =========================
# FOOTER
# =========================

st.markdown("---")

st.write("Developed for BITS Machine Learning Assignment 2")
