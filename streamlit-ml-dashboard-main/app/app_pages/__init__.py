# Import all page classes using importlib to handle emoji filenames
import importlib
import sys
import os

# Add the app_pages directory to Python path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import classes from emoji-named files
project_summary_module = importlib.import_module('1_00_📘_Project_Summary')
ProjectSummaryPage = project_summary_module.ProjectSummaryPage

eda_module = importlib.import_module('1_01_🔎_EDA')
EdaPage = eda_module.EdaPage

train_module = importlib.import_module('1_02_🧠_Train_Model')
TrainModelPage = train_module.TrainModelPage

predict_module = importlib.import_module('1_03_📈_Predict')
PredictPage = predict_module.PredictPage

metrics_module = importlib.import_module('1_04_🧪_Model_Metrics')
MetricsPage = metrics_module.MetricsPage

__all__ = [
    'ProjectSummaryPage',
    'EdaPage', 
    'TrainModelPage',
    'PredictPage',
    'MetricsPage'
]
