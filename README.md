
# 📦 Supply Chain Forecasting Pipeline

This repository contains all the necessary files, scripts, and documentation to run and maintain a demand forecasting system using LightGBM and the `skforecast` library. It includes the full pipeline for both model training and prediction, as well as a dashboard built with Streamlit for visualization and interaction.

The streamlit dashboard app was built on top of a template, forked from: [GitHub - freewimoe/streamlit-ml-dashboard](https://github.com/freewimoe/streamlit-ml-dashboard)

#### **How to Run the Dashboard**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the app**
   ```bash
   streamlit run app/app.py
   ```

3. **Open in browser**
   Navigate to: [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Forecasting Details

- **Forecasting Library**: [Skforecast](https://skforecast.org/latest/)
- **Model**: LightGBM
- **Forecasting Horizon**: Adjustable, with examples using H=4 and H=8 weeks

---

## 📁 Repository Structure

```text
├── data/
│   ├── master_file.xlsx          # Fully preprocessed dataset (2024-07-03 to 2025-07-30), including exogenous variables. Ready for direct model input.
│   └── master_file.csv           # Same dataset in CSV format. Required by the Streamlit application.
│
├── initial notebooks/                     # Contains data, notebooks, scripts, and modules used in the modeling pipeline.
│   ├── Data_processing.ipynb     # Initial data exploration and PSR file validation.
│   ├── Univariate_forecasting.ipynb  # Early data engineering using raw PSR files (2024-07-03 to 2025-07-30).
│   ├── Forecasting_normalized_h4.ipynb  # Full training and prediction workflow (non-modular version).
│   ├── Resources_split.ipynb     # Computes split values from forecasted ADD (first approach).
│
├── modules/                      # Python modules for reusable functions across the forecasting pipeline.
│   ├── Features.py               # Data loading, preprocessing, normalization, and train-test split.
│   ├── Lgbm_architecture.py      # LightGBM model builder, backtesting, tuning, and evaluation.
│   ├── Metrics.py                # Evaluation metrics and utilities for rescaling results.
│   └── Plotting.py               # Visualization functions for model results and performance metrics.
│
│
├── Model_v1.ipynb            # Implements forecasting using modular code.
├── Global_model.py           # Standalone script to train and forecast using defined parameters and forecast horizon.
│
│
└── streamlit-ml-dashboard-main/  # Streamlit app for interactive model usage (from GitHub template)
    ├── app/
    │   ├── app.py                # Main entry point to launch the dashboard
    │   ├── app_pages/            # Custom dashboard pages
    │   │   ├── __init__.py
    │   │   ├── 1_00_📘_Project_Summary.py
    │   │   ├── 1_01_🔎_EDA.py
    │   │   ├── 1_02_🧠_Train_Model.py
    │   │   ├── 1_03_📈_Predict.py
    │   │   └── 1_04_🧪_Traceability.py
    │   └── models/
    │       └── versioned/
    │           └── v1/
    │               └── latest.joblib  # Exported LightGBM model for prediction

```
