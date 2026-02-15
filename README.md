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

Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
logistic_regression	0.956140350877193	0.9977071732721913	0.9459459459459459	0.9859154929577465	0.9655172413793104	0.9068106119605033
decision_tree	0.9298245614035088	0.9253193580085163	0.9436619718309859	0.9436619718309859	0.9436619718309859	0.8506387160170324
knn	0.956140350877193	0.9959056665574845	0.9342105263157895	1.0	0.9659863945578231	0.9086150974691304
naive_bayes	0.9736842105263158	0.9983622666229938	0.9594594594594594	1.0	0.9793103448275862	0.9447329926514414
random_forest	0.9649122807017544	0.9962332132328856	0.958904109589041	0.9859154929577465	0.9722222222222222	0.9252853920667758
xgboost	0.956140350877193	0.9908286930887652	0.9583333333333334	0.971830985915493	0.965034965034965	0.9063785942932301

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

## Sample Test File
A sample input file `test_prediction.csv` is provided in this repository.
Use this file to test the prediction functionality in the Streamlit app.
Steps:
1. Run the app using: streamlit run app.py
2. Upload test_prediction.csv
3. View prediction results

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
