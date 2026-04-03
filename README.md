# 💳 Credit Risk Prediction System

A machine learning project to predict the **creditworthiness of loan applicants** using financial and behavioral data. This system helps in identifying whether a customer is likely to **default or repay** a loan.

---

## 🚀 Features

* 📊 Data preprocessing and cleaning pipeline
* ⚙️ Feature engineering & WOE binning
* 🤖 Machine Learning model for credit risk prediction
* 📈 Model evaluation metrics
* 🌐 API built using FastAPI
* 🖥️ Interactive UI using Streamlit

---

## 🏗️ Project Structure

```
CREDIT-RISK/
│
├── api/                  # FastAPI backend
│   └── app.py
│
├── src/                  # Core ML logic
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── woe_binning.py
│   ├── scoring.py
│   └── train_scorecard.py
│
├── notebooks/            # Jupyter notebooks (EDA)
│   └── eda.ipynb
│
├── streamlit_app.py      # Streamlit UI
├── main.py               # Entry point
├── train.py              # Model training
├── evaluate.py           # Model evaluation
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🧠 Tech Stack

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Streamlit

---

## ⚙️ Installation

```bash
git clone https://github.com/Karthikeya070/credit-risk.git
cd credit-risk
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1️⃣ Run FastAPI server

```bash
uvicorn api.app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

### 2️⃣ Run Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Workflow

1. Data Cleaning
2. Feature Engineering
3. WOE Binning
4. Model Training
5. Evaluation
6. Deployment via API/UI

---

## ⚠️ Note

* Large datasets and model files are excluded using `.gitignore`
* You can use your own dataset for training

---

## 📌 Future Improvements

* Hyperparameter tuning
* Model explainability (SHAP/LIME)
* Deployment on cloud (AWS/GCP)
* CI/CD pipeline

---

## 🤝 Contributing

Feel free to fork this repo and submit pull requests!

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Karthikeya**
GitHub: https://github.com/Karthikeya070

---
