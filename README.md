# 🏦 Bank Customer Churn Prediction

An end-to-end Machine Learning project designed to predict bank customer churn and identify customers who are likely to leave the bank.

## 📌 Project Overview

Customer churn is a major challenge for banking institutions. Retaining existing customers is essential for sustainable business growth.

This project uses Machine Learning techniques to predict customer churn, analyze customer behavior, and support data-driven customer retention strategies.

## 🎯 Project Objectives

- Predict whether a bank customer is likely to leave.
- Identify important factors influencing customer churn.
- Perform Exploratory Data Analysis (EDA).
- Build and evaluate Machine Learning models.
- Generate customer churn predictions using a trained model.
- Support proactive customer retention strategies.

## 📂 Project Structure

```text
Bank-Customer-Churn-Prediction/
│
├── dataset/
├── datasets/
│   └── processed_data/
│
├── models/
│   ├── best_random_forest.pkl
│   ├── encoder.pkl
│   ├── feature_column.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
│
├── app.py
├── requirements.txt
└── .gitignore
```

## 🔬 Project Workflow

1. **Exploratory Data Analysis:** Understanding customer data and identifying patterns.
2. **Data Preprocessing:** Cleaning and preparing data for Machine Learning.
3. **Feature Engineering:** Preparing relevant features for model training.
4. **Model Training:** Training and evaluating Machine Learning models.
5. **Model Selection:** Saving the selected Random Forest model.
6. **Prediction:** Using the trained model to predict customer churn.

## 🤖 Machine Learning

The project includes a trained Random Forest model.

Saved model artifacts include:

- Random Forest model
- Encoder
- Feature column information
- Scaler

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

## ⚙️ Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Mohammad-Taufique/Bank--Customer--Churn-Prediction.git
```

Navigate to the project directory:

```bash
cd Bank--Customer--Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Results

The project includes a trained Random Forest model and saved preprocessing artifacts.

Detailed evaluation metrics and model performance results will be documented after final validation.

## 🚀 Future Improvements

- Improve model performance through hyperparameter tuning.
- Add explainable AI using SHAP.
- Enhance customer risk scoring.
- Improve the prediction application interface.
- Deploy the application online.

## 👨‍💻 Author

**Mohammad Taufique**

GitHub: [Mohammad-Taufique](https://github.com/Mohammad-Taufique)

---

⭐ If you find this project interesting, consider giving the repository a star!
