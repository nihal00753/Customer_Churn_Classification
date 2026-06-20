# Customer Churn Prediction (ANN)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Sequential%20ANN-d00000)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Preprocessing-f7931e)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A binary classifier that predicts whether a bank customer will churn (exit), built with a feed-forward Artificial Neural Network in TensorFlow/Keras and deployed as an interactive Streamlit app.

## Live Demo

https://customerchurnclassification0753.streamlit.app

## Overview

- Predicts churn probability from 10 customer attributes (credit score, geography, gender, age, tenure, balance, products held, credit card status, activity status, salary)
- Full pipeline: data cleaning, encoding, scaling, ANN training with early stopping, and a Streamlit inference app
- Built as a hands-on project to apply deep learning fundamentals (Sequential API, callbacks, TensorBoard) on a structured/tabular dataset

## Tech Stack

- **Modeling:** TensorFlow, Keras
- **Preprocessing:** scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler)
- **Data handling:** pandas, NumPy
- **App/Deployment:** Streamlit
- **Experiment tracking:** TensorBoard

## Dataset

- `Churn_Modelling.csv` — 10,000 bank customers, 14 columns
- Target: `Exited` (1 = churned, 0 = retained)
- Dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`) as non-predictive
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

## Pipeline

1. Drop irrelevant identifier columns
2. Label-encode `Gender`
3. One-hot encode `Geography` (France / Germany / Spain)
4. Train/test split (80/20, `random_state=42`)
5. Standard-scale all features
6. Persist `LabelEncoder`, `OneHotEncoder`, and `StandardScaler` as `.pkl` files for reuse at inference time

## Model Architecture

| Layer | Units | Activation |
|---|---|---|
| Input | 12 features | — |
| Dense (Hidden 1) | 64 | ReLU |
| Dense (Hidden 2) | 32 | ReLU |
| Dense (Output) | 1 | Sigmoid |

- **Optimizer:** Adam (learning rate = 0.01)
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping (monitor=`val_loss`, patience=10, restores best weights), TensorBoard logging
- **Max epochs:** 100 (training stopped early at epoch 17, best weights restored from epoch 7)

## Results

| Metric | Train | Validation |
|---|---|---|
| Accuracy | ~86.0% | ~85.7% |
| Loss | ~0.323 | ~0.334 |

Best-performing weights (lowest validation loss) were automatically restored by early stopping.

## Project Structure

```
Customer_Churn_Classification/
├── Churn_Modelling.csv          # Source dataset
├── experiments.ipynb            # Preprocessing + ANN training
├── prediction.ipynb             # Inference walkthrough
├── app.py                       # Streamlit app
├── model.h5                     # Trained Keras model
├── label_encoder_gender.pkl     # Saved gender encoder
├── onehot_encoder_geo.pkl       # Saved geography encoder
├── scaler.pkl                   # Saved feature scaler
├── requirements.txt
└── README.md
```

## Setup / Run Locally

```bash
git clone https://github.com/nihal00753/Customer_Churn_Classification.git
cd Customer_Churn_Classification
pip install -r requirements.txt
streamlit run app.py
```

## Possible Improvements

- Pin dependency versions in `requirements.txt` for reproducible deploys
- Add evaluation metrics beyond accuracy (precision, recall, F1, ROC-AUC) given class imbalance is common in churn data
- Cache model/encoder loading in the Streamlit app (`@st.cache_resource`)
- Compare ANN performance against a simpler baseline (e.g. Logistic Regression, XGBoost)
