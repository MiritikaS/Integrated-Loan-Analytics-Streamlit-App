# Integrated Loan Analytics Streamlit App

This app contains:

1. EDA Analysis summary
2. Power BI dashboard screenshots and insights
3. Loan Approval Recommendation tool

## Files

```text
train_loan_approval_model.py
integrated_loan_analytics_app.py
requirements.txt
assets/
  loan_portfolio_risk_summary.png
  product_credit_behaviour.png
model_outputs/        created after training
```

## How To Run

Open Command Prompt:

```bat
cd "PATH_TO_THIS_FOLDER"
```

Install packages:

```bat
pip install -r requirements.txt
```

Train model:

```bat
python train_loan_approval_model.py
```

Run app:

```bat
python -m streamlit run integrated_loan_analytics_app.py
```

## Login

```text
User ID: faculty
Password: loananalytics123
```

## Presentation Explanation

Say:

"The Streamlit app has three sections. The first section shows EDA analysis summary, the second section shows Power BI dashboard screenshots and insights, and the third section is the Loan Approval Recommendation app. The recommendation app uses a Logistic Regression model and business rules to suggest Approve, Manual Review, or Reject / Very Strict Review."

