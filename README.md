A machine learning project that predicts whether a person is likely to default on a loan based on financial history.
Deployed as an interactive web app using Gradio on Hugging Face Spaces.
👉 Live Demo:
https://huggingface.co/spaces/Niaj-Morshed-Niloy/Credit-Risk-Prediction-System

Problem Statement:
Financial institutions need to evaluate whether a loan applicant is high-risk or low-risk.
The goal of this project is to build a classification model that predicts credit default risk using historical financial data.

Dataset Name: Give Me Some Credit (Kaggle)
It contains financial attributes such as:Income,Debt ratio,Number of past due payments,Credit utilization,Number of credit lines,Dependents.

Machine Learning Workflow
1. Data Preprocessing
Handled missing values using median imputation
Feature selection from financial attributes
Converted dataset into model-ready format

3. Handling Class Imbalance
The dataset was highly imbalanced.
Solutions applied:
class_weight='balanced'
Threshold tuning

5. Models Used
Random Forest Classifier
Better performance on non-linear patterns
Key settings:
n_estimators=200
class_weight='balanced'
max_depth=10

6. Web App (Gradio Deployment)
Users input financial details such as:
Debt ratio
Income
Past due history
Credit utilization
The model outputs:
⚠️ High Risk of Default
OR
✅ Low Risk of Default
with a probability score.

Author:
Md. Niaj Morshed
CS Student| AI/ML Enthusiast | Python Developer
