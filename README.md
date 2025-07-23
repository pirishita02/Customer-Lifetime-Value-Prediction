# 💰 Customer Lifetime Value Prediction Web App

🔗 Example Usage Video

🎥 Watch how to use the app from upload to prediction in this demo video:

https://github.com/user-attachments/assets/0317031f-44be-4945-8855-cae40e5d5cc1


A fully interactive and deployable web application built using **Streamlit** that predicts **Customer Lifetime Value (CLV)** from transactional data. It combines real-time dataset ingestion, advanced EDA, preprocessing, machine learning modeling, and prediction capabilities in one intuitive dashboard. 

---

## 🚀 Features

### ✅ 1. Dataset Upload & Ingestion
- Supports uploading **CSV files** directly via the app interface.
- Automatically detects or engineers key CLV-related features such as:
  - `CustomerID`
  - `Recency`, `Frequency`, `Monetary Value`
  - `First Purchase Date`, `Last Purchase Date`
  - `Average Purchase Value`
  - `Tenure` (time since first purchase)

### 📊 2. Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, median, std, etc.)
- Visualizations:
  - Purchase frequency distributions
  - Recency vs. Frequency plots
  - Monetary distribution histograms
  - Outlier detection visuals
- Insights on variable correlations with predicted CLV

### 🧹 3. Data Preprocessing
- Handles:
  - Missing values
  - Outlier capping
  - Label encoding for categorical features
  - Standard scaling for numeric columns
- Option to **download the cleaned dataset** directly

### 🤖 4. Machine Learning Modeling
- Predicts **Customer Lifetime Value (CLV)** 
- Evaluation Metrics:
  - **MAE**, **RMSE**, **R² Score**
- Train or retrain on uploaded datasets in real-time

### 🧩 5. Dynamic UI Components
- Upload your custom dataset
- View EDA summary and insights
- Select and train ML model of your choice
- Visualize prediction performance
- Make predictions on new customer entries
- Download predicted results

### ☁️ 6. Deployment Ready
- Fully compatible with **Streamlit Cloud**
- Includes `requirements.txt` and `streamlit_app.py` for easy reproduction
- Lightweight and responsive layout

### 💡 7. Extras (Polish Features)
- 📘 “What is CLV?” section for non-technical users
- 📈 Prediction intervals or confidence ranges
- 📝 Textual insights embedded beside charts for better interpretation
