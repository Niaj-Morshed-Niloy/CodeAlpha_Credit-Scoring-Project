import gradio as gr
import pandas as pd
import pickle

with open("CreditScore.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
scaler = data["scaler"]

def predict_credit(
    revolving_utilization,
    age,
    past_due_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    past_due_90,
    real_estate_loans,
    past_due_60_89,
    dependents
):

    input_data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": past_due_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
        "NumberOfDependents": dependents
    }])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    threshold = 0.42

    if prob >= threshold:
        return f"High Risk of Default (Risk Score: {prob:.2f})"
    else:
        return f"Low Risk of Default (Risk Score: {prob:.2f})"

interface = gr.Interface(
    fn=predict_credit,
    inputs=[
        gr.Number(label="Revolving Utilization"),
        gr.Number(label="Age"),
        gr.Number(label="30-59 Days Past Due"),
        gr.Number(label="Debt Ratio"),
        gr.Number(label="Monthly Income"),
        gr.Number(label="Open Credit Lines"),
        gr.Number(label="90+ Days Past Due"),
        gr.Number(label="Real Estate Loans"),
        gr.Number(label="60-89 Days Past Due"),
        gr.Number(label="Number of Dependents")
    ],
    outputs="text",
    title="Credit Risk Prediction System",
    description="Predicts whether a person is high or low risk for loan default using financial history data."
)

interface.launch()