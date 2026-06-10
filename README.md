# 🛡️ Subscriber Retention AI

An end-to-end Machine Learning platform designed to predict customer churn, uncover retention opportunities, and support data-driven business decisions.

The application transforms raw subscription data into actionable insights through predictive insights, interactive dashboards, and retention strategy simulations.

## 🚀 Live Demo

https://airabhiii-subscriber-retention-ai-app-yma1vb.streamlit.app/

---

## ✨ Features

### 📊 Executive KPI Dashboard

* Monitor churn rate, customer count, and revenue at risk.
* Quickly identify high-risk customer segments.

### 🤖 Predictive Insights

* Machine Learning-powered prediction.
* Uses advanced classification techniques for improved performance.

### 🎯 Retention Strategy Simulator

* Test discount and retention offers.
* Observe how different strategies impact churn probability.

### 💡 Intelligent Insights

* Automated recommendations to support retention decisions.
* Highlights factors contributing to customer churn.

### ⚖️ Threshold Optimization

* Adjust prediction thresholds.
* Explore Precision vs Recall trade-offs for business needs.

---

## 📁 Project Structure

```text
Subscriber-Retention-AI/
│
├── app.py
├── requirements.txt
├── data/
│   └── churn.csv
└── README.md
```

---

## 🛠️ Tech Stack

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Machine Learning     | Scikit-learn, SMOTE |
| Visualization        | Matplotlib, Seaborn |
| Web Application      | Streamlit           |

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AirAbhiii/Subscriber-Retention-AI.git
cd Subscriber-Retention-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 📈 Dataset

The application can generate realistic synthetic subscriber data, allowing users to explore the platform without requiring an external dataset.

To use your own dataset:

1. Create a `data` folder.
2. Add a `churn.csv` file.
3. Ensure the dataset contains a target column named `Churn`.

---

## 📋 Application Modules

| Module                   | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| Data Overview            | Explore customer data and churn distribution          |
| Feature Engineering      | Understand engineered features and feature importance |
| Imbalanced Data Analysis | Learn how class imbalance affects model performance   |
| Threshold Tuning         | Optimize classification thresholds                    |
| Churn Prediction         | Predict churn risk and simulate retention strategies  |

---

## 🎯 Project Goals

* Predict customer churn using Machine Learning.
* Visualize key business metrics.
* Demonstrate handling of imbalanced datasets.
* Support customer retention decision-making.
* Provide an interactive analytics experience through Streamlit.





