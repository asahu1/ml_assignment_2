import streamlit as st
import pandas as pd
import joblib
import os
import pickle

# -----------------------------
# App Title
# -----------------------------
st.title("BITS ML Assignment 2 - Classification Models")
st.write("Comparison of 6 Machine Learning Models")

# -----------------------------
# Models Folder
# -----------------------------
MODELS_FOLDER = "models"
results_path = os.path.join(MODELS_FOLDER, "results.csv")
feature_path = os.path.join(MODELS_FOLDER, "feature_names.pkl")

# -----------------------------
# Check results file
# -----------------------------
if os.path.exists(results_path):

    results = pd.read_csv(results_path)

    st.subheader("Model Performance Metrics")
    st.dataframe(results)

    # -----------------------------
    # Model Selection
    # -----------------------------
    model_name = st.selectbox(
        "Select Model for Prediction",
        results["Model"]
    )

    model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        # -----------------------------
        # Load feature names
        # -----------------------------
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                feature_names = pickle.load(f)
        else:
            st.error("feature_names.pkl not found. Please retrain and save features.")
            st.stop()

        # -----------------------------
        # Download Sample CSV
        # -----------------------------
        st.subheader("Download Sample CSV Format")

        sample_df = pd.DataFrame(columns=feature_names)

        st.download_button(
            label="Download Sample Input CSV",
            data=sample_df.to_csv(index=False),
            file_name="sample_input.csv",
            mime="text/csv"
        )

        # -----------------------------
        # Upload CSV for Prediction
        # -----------------------------
        st.subheader("Upload CSV File for Prediction")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:

            try:
                data = pd.read_csv(uploaded_file)

                # Ensure correct columns
                data = data[feature_names]

                predictions = model.predict(data)

                st.success("Prediction Successful!")

                result_df = data.copy()
                result_df["Prediction"] = predictions

                st.subheader("Prediction Results")
                st.dataframe(result_df)

            except KeyError:
                st.error("Uploaded CSV does not have required feature columns.")
            except Exception as e:
                st.error("An error occurred during prediction.")
                st.write("Error details:", str(e))

    else:
        st.error("Selected model file not found in models folder.")

else:
    st.error("results.csv not found in models folder.")
