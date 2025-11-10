
# 📦 Supply Chain Forecasting Pipeline

This repository contains all the necessary files, scripts, and documentation to run and maintain a demand forecasting system using LightGBM and the `skforecast` library. It includes the full pipeline for both model training and prediction, as well as a dashboard built with Streamlit for visualization and interaction.

---

## 📁 Repository Structure

### 🔹 `Programs/`

Contains the data, notebooks, scripts, and modules used throughout the modeling pipeline.

#### **Preprocessed Data Files**
- **`Features_py.xlsx`**: Fully preprocessed dataset from `2024-07-03` to `2025-07-30`, including exogenous variables. Ready for direct model input.
- **`Features_py.csv`**: Same as above in CSV format. Required by the Streamlit application.

#### **Initial Notebooks**
- **`Data_processing.ipynb`**: Initial data exploration and validation of the PSR files. Identifies header inconsistencies due to data format updates.
- **`Univariate_forecasting.ipynb`**: Early data engineering steps using raw PSR files from `data/` folder. Time range: `2024-07-03` to `2025-07-30`.
- **`Forecasting_normalized_h4.ipynb`**: Full training and prediction workflow in notebook format (without using external modules). Covers the **first approach** (`series = Product ID`).
- **`Resources_split.ipynb`**: Follow-up notebook to compute split values from forecasted ADD. Completes the **first approach**.
- **`Resources_split_M1.ipynb`**: Same process as above but implements the **second approach** (`series = Brand||Resource ID`), directly forecasting split values.

#### **Optimized Notebooks (using modules)**
- **`Model_v1.ipynb`**: Implements **first approach** using modular code.
- **`Model_v2.ipynb`**: Implements **second approach** using modular code.
- **`Model_v1_h8.ipynb`** : Trains and predicts using a horizon of 8 weeks. Useful for side-by-side comparison of outputs.
- **`Model_v2_h8.ipynb`** : Trains and predicts using a horizon of 8 weeks.

> 🔹 *First approach*: forecast ADD → derive split values  
> 🔹 *Second approach*: directly forecast split values

#### **Other Python Scripts**
- **`Global_model.py`**: Standalone script to execute the full training and forecasting pipeline using selected parameters and forecast horizon (`steps`).

---

### 🔹 `Modules/`

This folder contains Python modules for reusable functions across the forecasting pipeline.

- **`Features.py`** *(typo: consider renaming to `Features.py`)*: Functions for data loading, preprocessing, normalization, and train-test split.
- **`Lgbm_architecture.py`**: Functions to build and train LightGBM forecaster, including backtesting, tuning, and evaluation.
- **`Metrics.py`**: Evaluation metrics (MAE, absolute error, etc.) and utilities for converting scaled results back to original scale.
- **`Plotting.py`**: Visualization functions for model results and metrics.

#### **Utilities**
- **`Series_status.py`**: Script to generate the `series_status.xlsx`/`.csv` file. Tracks series lifecycle (appearance/disappearance), useful for production traceability.

---

## 📊 Streamlit Dashboard

### 🔸 Location: `streamlit-ml-dashboard-main/`

Forked from:  
[GitHub - freewimoe/streamlit-ml-dashboard](https://github.com/freewimoe/streamlit-ml-dashboard)

#### **Customizations**
- Added `Modules/` folder for backend functionality.
- Created custom pages:
  - `Train_Model_Copy`
  - `Predict_Copy`
- **Project Summary** and **Traceability** pages need to be updated:
  - *Project Summary:* should include updated app usage instructions.
  - *Traceability:* intended to display `series_status` table and historic graphs to monitor model in production.

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
- **Custom GPT**: A ChatGPT agent has been fine-tuned for working with Skforecast. (Access may require permission) [ChatGPT - Skforecast Helper](https://chatgpt.com/g/g-68a638426e5881918532c83e4472be23-skforecast-helper)

---

## 📌 Next Steps & Automation Plan

The following steps are planned to automate the entire pipeline from data ingestion to forecast generation:

1. **Automate daily data ingestion**
   - Create a script to clean and preprocess Excel files received daily from the OAC platform via email.
   - Coordinate with **Brian** to redirect those emails to **Sanjana**.

2. **Integrate with Power Automate**
   - Combine preprocessing with Power Automate flow.
   - Collaborate with **Tanner**, **Ilicia**, and **Brian** for implementation.

3. **Unify ingestion + forecasting**
   - Link automated preprocessing to model training and prediction.
   - Aim to automate the full cycle dynamically via the Streamlit app.

4. **Build traceability system**
   - Develop a graph or logging mechanism to track forecast history and evaluate performance over time.

---

## 📁 Project Structure

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
│
│   ├── data/raw/                     # Sample data from template (not used)
│   │   ├── sample_house_prices.csv
│   │   └── sample_iris.csv
│
│   └── .spectory/, streamlit/        # Supporting folders from the original template
│
└── venv_psr/                          # Python virtual environment for the project (local dependencies)
