# 🛡️ Subscriber Retention AI

An end-to-end Machine Learning platform designed to predict, analyze, and mitigate customer churn. This project transforms raw subscription data into actionable business intelligence using advanced classification models and real-time retention simulation tools.

## 🚀 Key Features

- **Executive KPI Dashboard:** Instant visibility into business health, including Total Customers, Churn Rate, and Monthly Revenue at Risk.
- **Advanced Churn Modeling:** Employs Gradient Boosting paired with SMOTE to effectively handle imbalanced datasets and drastically improve churn detection (Recall).
- **Retention Offer Simulator:** An interactive tool that allows stakeholders to simulate discount strategies and visualize their impact on churn risk in real-time.
- **AI-Powered Insights:** Provides automated business-logic reasoning for churn predictions, helping teams decide between simple discounts or strategic contract upgrades.
- **Decision Threshold Tuning:** A professional utility to balance Precision and Recall, enabling users to optimize the model based on actual business costs.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** pandas, NumPy
- **Machine Learning:** scikit-learn, imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib, Seaborn
- **UI/Framework:** Streamlit

## 🏃 How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the application: `streamlit run app.py`

## 📊 Data Source

The app defaults to generating realistic synthetic subscription data for demonstration. To use your own data, place a `churn.csv` file in the `data/` directory.
