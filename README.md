# 🐟 Fish Weight Prediction App

A complete end-to-end Machine Learning project covering:

- 📊 Exploratory Data Analysis (EDA)
- 🛠 Feature Engineering
- 🤖 Model Building
- 📈 Model Evaluation
- 🌐 Streamlit Deployment

---

## 🚀 Live Demo

👉 **Streamlit App:**  
https://fish-project-deploy-z2e5mh4hrasp6whhg8fq3k.streamlit.app/

---

## 📌 Project Overview

This project predicts fish weight (in grams) using physical measurements and engineered features.

The model applies:

- Log transformation (for skew correction)
- Feature engineering (volume proxy, polynomial terms, interaction terms)
- Species encoding (One-Hot Encoding)
- Linear Regression (trained on log-transformed target)

The final predictions are back-transformed using exponential function to return accurate weight in grams.

---

## 📊 Dataset Information

The dataset contains 476 fish samples across 7 species with the following features:

- Species
- Length1 (cm)
- Height (cm)
- Width (cm)
- Girth (cm)
- Weight (g)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

## 🔄 Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Log Transformation of Target
5. Feature Engineering
   - Volume Proxy
   - Log Volume
   - Polynomial Features
   - Interaction Terms
6. Species Encoding (get_dummies)
7. Model Training (Linear Regression)
8. Model Evaluation (RMSE, MAE, R²)
9. Model Saving (.pkl)
10. Streamlit Deployment

---

## 📁 Project Structure

fish_project/
│
├── app.py
├── Fish_Assessment.ipynb
├── Linear\_\_1_assement.csv
├── fish_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md

---

## 📈 Model Performance

Evaluation Metrics Used:

- R² Score
- RMSE
- MAE

The log-transformed model showed improved residual distribution and better generalization.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
