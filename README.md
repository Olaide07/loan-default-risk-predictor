# Loan Default Risk Predictor

An end-to-end machine learning project that predicts the likelihood of loan default using financial and credit-related features.

---

## Project Overview

This project builds a predictive model to identify high-risk borrowers using a real-world financial dataset. The focus was on improving the model’s ability to detect defaulters (recall), which is critical in credit risk systems.

---

## Key Objectives

* Predict whether a customer will default on a loan
* Handle class imbalance effectively
* Optimize model performance using threshold tuning
* Build an interactive web app for real-time predictions

---

## Key Results

* Improved recall from **3% → 77%**
* Reduced missed defaulters significantly
* Balanced model performance using threshold tuning (0.40)
* Built a Streamlit app for live predictions

---

## Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib

---

## How It Works

1. Data cleaning and preprocessing
2. Feature selection and engineering
3. Model training using Random Forest and XGBoost
4. Handling class imbalance
5. Threshold tuning to improve recall
6. Deployment via Streamlit

---

## Live Demo

[Click here to try the app](https://loan-default-risk-predictor.streamlit.app/)

```

## Model Performance

The model was optimized for recall to minimize false negatives (missed defaulters), which is crucial in financial risk systems.

| Metric                    | Value |
| ------------------------- | ----- |
| Recall (Default Class)    | 0.77  |
| Precision (Default Class) | 0.29  |
| Accuracy                  | 0.58  |

---

## Key Insight

Instead of optimizing for accuracy, the model prioritizes **recall**, ensuring that most high-risk customers are correctly identified.

---

## Streamlit App

The project includes an interactive web app where users can:

* Input customer details
* Adjust decision threshold
* View prediction probability
* Explore feature importance

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Olaide07/loan-default-risk-predictor.git
cd loan-default-risk-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:
streamlit run app.py

```
Project Structure
loan-default-risk-predictor/
│
├── app.py
├── xgb_model.pkl
├── requirements.txt
├── README.md
```
```
Future Improvements
Improve precision while maintaining high recall
Add SHAP for model explainability
Deploy model with full preprocessing pipeline
Integrate real-time data sources

Author
Built by Olaide Ajibade

Acknowledgment
This project demonstrates practical machine learning skills including data preprocessing, model evaluation, and deployment.

