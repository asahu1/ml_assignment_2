Machine Learning Assignment 2 – Model Training & Deployment

Student Name: Anshul Sahu
Student ID : 2025AA05906
Student Mail ID: 2025aa05906@wilp.bits-pilani.ac.in
Course: Machine Learning
Environment Used: BITS Lab

Project Overview

This project demonstrates the complete machine learning lifecycle including:
Data preprocessing
Training multiple ML models
Performance comparison
Saving trained models
Deploying the best model using Streamlit
The final application allows users to upload a CSV file and receive predictions through a web interface.

Models Implemented
The following machine learning models were trained and evaluated:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier

Each model was trained on the training dataset and evaluated using test data.

Model Evaluation

Performance metrics such as accuracy were used to compare models.
The results are stored in:
models/results.csv

Models Used and Performance Comparison

The following 6 models were implemented:
Logistic Regression
Decision Tree
k-Nearest Neighbors (kNN)
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)

Performance Comparison Table

Model Name Accuracy AUC Precision Recall F1 Score MCC 
logistic_regression 0.95614 0.997707 0.945946 0.985915 0.965517 0.906811
decision_tree 0.929825 0.925319 0.943662 0.943662 0.943662 0.850639
knn 0.95614 0.995906 0.934211 1 0.965986 0.908615
naive_bayes 0.973684 0.998362 0.959459 1 0.97931 0.944733
random_forest 0.964912 0.995578 0.958904 0.985915 0.972222 0.925285
xgboost 0.95614 0.990829 0.958333 0.971831 0.965035 0.906379

All metrics were calculated using the test dataset.
Where:
Accuracy = Overall correctness
AUC = Area under ROC curve
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = Harmonic mean of precision and recall
MCC = Matthews Correlation Coefficient

Observations on Model Performance

ML Model Name - Observation about Model Performance

Logistic Regression
Performed very well with high AUC (0.9977). Indicates strong linear separability in the dataset. Stable and interpretable model.

Decision Tree
Lowest performance among all models. Slightly lower MCC indicates possible overfitting and weaker generalization.

kNN
Achieved perfect Recall (1.0), meaning it correctly identified all positive samples. Performance depends on distance metric and scaling.

Naive Bayes
Best performing model overall with highest Accuracy (0.9737), AUC (0.9984), F1 (0.9793), and MCC (0.9447). Shows strong probabilistic classification performance.

Random Forest (Ensemble)
Very strong and stable performance. Ensemble method reduces overfitting and improves generalization.

XGBoost (Ensemble)
Good performance but slightly lower than Random Forest and Naive Bayes on this dataset. Still robust and powerful boosting model.

Conclusion
Naive Bayes achieved the best overall performance.
Ensemble models (Random Forest & XGBoost) also performed strongly.
Decision Tree showed comparatively lower performance.
The dataset appears well-separated, given very high AUC values across models.

Project Structure
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

Installation

Clone the repository:

git clone https://github.com/asahu1/ml_assignment_2.git
cd ml_assignment_2

Install required libraries:

pip install -r requirements.txt

Running the Jupyter Notebook

To train the models:

jupyter notebook ml_assignment_2.ipynb

This notebook will:

Load dataset
Preprocess data
Train models
Evaluate performance
Save trained models in models/ folder
Save performance metrics in results.csv

Running the Streamlit App (Local)

Run the following command:
streamlit run app.py

Then open in browser:
http://localhost:8501

Upload a CSV file to generate predictions.

Deployed Application

The application is deployed using Streamlit Community Cloud.

Live App Link:
https://mlassignment2-3orp7qh8i3zcyri8vdmnyv.streamlit.app/

GitHub Repository

Repository Link:
https://github.com/asahu1/ml_assignment_2

Technologies Used

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

 GitHub Repository
 Deployed Streamlit App
 Jupyter Notebook
 Trained Models
 Results CSV
 Final Submission PDF
