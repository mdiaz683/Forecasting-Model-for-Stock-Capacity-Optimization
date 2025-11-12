
# 📦 Supply Chain Forecasting Pipeline

This repository contains all the necessary files, scripts, and documentation to run and maintain a demand forecasting system using LightGBM and the `skforecast` library. It includes the full pipeline for both model training and prediction, as well as a dashboard built with Streamlit for visualization and interaction.

---

## 📁 Repository Structure

### 🔹 `Data/`
- **`master_file.xlsx`**: Fully preprocessed dataset from `2024-07-03` to `2025-07-30` (weekly snapshots), including exogenous variables. Ready for direct model input.
- **`master_file.csv`**: Same as above in CSV format. Required by the Streamlit application.
## 📄 File Descriptions

| File                   | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `master_file.xlsx`     | Fully preprocessed dataset from `2024-07-03` to `2025-07-30` (weekly snapshots), including exogenous variables. Ready for direct model input.                            |
| `master_file.csv`      | Same as above in CSV format. Required by the Streamlit application.                                                                     |

### 🔹 `Programs/`
Contains the data, notebooks, scripts, and modules used throughout the modeling pipeline.
| File                   | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `Features.py`     | Functions for data loading, preprocessing, normalization, and train-test split.|
| `Lgbm_architecture.py`      | Functions to build and train LightGBM forecaster, including backtesting, tuning, and evaluation.|
| `Metrics.py`     | Evaluation metrics (MAE, absolute error, etc.) and utilities for converting scaled results back to original scale.|
| `Plotting.py` | Visualization functions for model results and metrics.|

#### **Initial Notebooks**
- **`Data_processing.ipynb`**: Initial data exploration and validation of the PSR files. Identifies header inconsistencies due to data format updates.
- **`Univariate_forecasting.ipynb`**: Early data engineering steps using raw PSR files from `data/` folder. Time range: `2024-07-03` to `2025-07-30`.
- **`Forecasting_normalized_h4.ipynb`**: Full training and prediction workflow in notebook format (without using external modules).
- **`Resources_split.ipynb`**: Follow-up notebook to compute split values from forecasted ADD. Completes the **first approach**.

#### **Optimized Files (using modules)**
| File                   | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `Model_v1.ipynb`     | Implements the forecasting using modular code..|
| `Global_model.py`      | Standalone script to execute the full training and forecasting pipeline using selected parameters and forecast horizon (`steps`).|

### 🔹 `Modules/`
This folder contains Python modules for reusable functions across the forecasting pipeline.

| File                   | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `Features.py`     | Functions for data loading, preprocessing, normalization, and train-test split.|
| `Lgbm_architecture.py`      | Functions to build and train LightGBM forecaster, including backtesting, tuning, and evaluation.|
| `Metrics.py`     | Evaluation metrics (MAE, absolute error, etc.) and utilities for converting scaled results back to original scale.|
| `Plotting.py` | Visualization functions for model results and metrics.|


---

## Streamlit Dashboard

### 🔸 Location: `streamlit-ml-dashboard-main/`
This app was built on top of a template, forked from: [GitHub - freewimoe/streamlit-ml-dashboard](https://github.com/freewimoe/streamlit-ml-dashboard)

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

## ⚙️ Technical Details

- **Forecasting Library**: [Skforecast](https://skforecast.org/latest/)
- **Model**: LightGBM
- **Forecasting Horizon**: Adjustable, with examples using H=4 and H=8 weeks

---

## 📁 Project Structure
## 📄 File Descriptions

| File                   | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `jde_app.py`           | Main script to launch the Streamlit app. Loads and processes Excel inputs.                            |
| `requirements.txt`     | List of required Python packages.                                                                     |
| `budget_plan.xlsx`     | Primary Excel data source for Budget vs Actuals (BvA), simulating an official corporate finance database. |
| `account_details.xlsx` | Detailed view of financials, per business unit or account.                                            |


Programs/
├── data/
│   ├── Clean Data/                    # Cleaned and formatted input files
│   ├── Raw Data/                      # Original unprocessed PSR input files
│   ├── master_csv.csv                 # Merged raw input (CSV format)
│   ├── master_excel.xlsx              # Merged raw input (Excel format)
│   ├── series_status.csv              # Tracking status of all series (CSV)
│   └── series_status.xlsx             # Tracking status of all series (Excel)
│
├── data - Copy/                       # Temporary backup or copy of data
│
├── modules/                           # Python modules for core functionality
│   ├── __init__.py                    # Declares this folder as a Python package
│   ├── features.py                    # Functions for loading and preprocessing data
│   ├── lgbm_architecture.py           # LightGBM forecaster construction and training
│   ├── metrics.py                     # Model evaluation metrics
│   └── plotting.py                    # Plotting and visualization utilities
│
├── results/                           # Processed and intermediate results
│   ├── features_py.csv
│   ├── features_py.xlsx
│   ├── series_status.csv
│   └── series_status.xlsx
│
├── data_processing.ipynb             # Initial data exploration and validation
├── forecasting_normalized_h4.ipynb   # First approach full pipeline (non-modular)
├── global_model.py                   # Full training + prediction script
├── model_v1_h8.ipynb                 # Modular pipeline - first approach, H=8
├── model_v1.ipynb                    # Modular pipeline - first approach
├── model_v2_h8.ipynb                 # Modular pipeline - second approach, H=8
├── model_v2.ipynb                    # Modular pipeline - second approach
├── README_SupplyChain.md             # Project documentation
├── resources_split.ipynb             # Forecast to split values (first approach)
├── resources_split_M1.ipynb          # Forecast directly split values (second approach)
├── series_status.py                  # Script to build the tracking table
├── univariate_forecasting.ipynb      # Early experimentation on raw PSR files
├── streamlit-ml-dashboard-main/      # Streamlit app for interactive model usage (from github template)
│   ├── app/
│   │   ├── app.py                     # Main entry point to launch the dashboard
│   │   ├── app_pages/                # Custom dashboard pages
│   │   │   ├── __init__.py
│   │   │   ├── 1_00_📘_Project_Summary.py
│   │   │   ├── 1_01_🔎_EDA.py
│   │   │   ├── 1_02_🧠_Train_Model copy.py
│   │   │   ├── 1_02_🧠_Train_Model.py
│   │   │   ├── 1_03_📈_Predict copy.py
│   │   │   ├── 1_03_📈_Predict.py
│   │   │   └── 1_04_🧪_Traceability.py
│   │   └── models/                   # Stores trained model objects
│   │       └── versioned/v1/
│   │           └── latest.joblib     # Exported LightGBM model for prediction

