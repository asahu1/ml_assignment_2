Machine Learning Assignment 2 – Model Training & Deployment
Project Overview

This project demonstrates the complete machine learning lifecycle including:

Data preprocessing

Training multiple ML models

Performance comparison

Saving trained models

Deploying the best model using Streamlit

The final application allows users to upload a CSV file and receive predictions through a web interface.

🧠 Models Implemented

The following machine learning models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Each model was trained on the training dataset and evaluated using test data.

📊 Model Evaluation

Performance metrics such as accuracy were used to compare models.

The results are stored in:

models/results.csv

Based on evaluation, Random Forest and XGBoost performed better compared to other models.

📂 Project Structure
ml_assignment_2/
│
├── ml_assignment_2.ipynb        # Model training notebook
├── app.py                       # Streamlit deployment script
├── requirements.txt             # Required libraries
├── README.md                    # Project documentation
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── results.csv
⚙️ Installation

Clone the repository:

git clone https://github.com/asahu1/ml_assignment_2.git
cd ml_assignment_2

Install required libraries:

pip install -r requirements.txt
📓 Running the Jupyter Notebook

To train the models:

jupyter notebook ml_assignment_2.ipynb

This notebook will:

Load dataset

Preprocess data

Train models

Evaluate performance

Save trained models in models/ folder

Save performance metrics in results.csv

🌐 Running the Streamlit App (Local)

Run the following command:

streamlit run app.py

Then open in browser:

http://localhost:8501

Upload a CSV file to generate predictions.

☁️ Deployed Application

The application is deployed using Streamlit Community Cloud.

🔗 Live App Link:
https://mlassignment2-3orp7qh8i3zcyri8vdmnyv.streamlit.app/

🔗 GitHub Repository

Repository Link:
https://github.com/asahu1/ml_assignment_2

📌 Technologies Used

Python

Pandas

Scikit-learn

XGBoost

Streamlit

Jupyter Notebook

Conclusion

This project successfully demonstrates:

Implementation of multiple ML algorithms
Model comparison and evaluation
Saving trained models using pickle
Deployment of ML model using Streamlit
End-to-end ML workflow from development to deployment
The project reflects real-world machine learning pipeline implementation.

Submission Includes

✔ GitHub Repository
✔ Deployed Streamlit App
✔ Jupyter Notebook
✔ Trained Models
✔ Results CSV
✔ Final Submission PDF
