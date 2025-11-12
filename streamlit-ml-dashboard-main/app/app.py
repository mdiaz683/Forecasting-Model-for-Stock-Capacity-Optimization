import streamlit as st
import sys
import os
import importlib.util

# Add current directory and src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
app_pages_dir = os.path.join(current_dir, 'app_pages')

for path in [current_dir, src_dir, app_pages_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import page classes using direct module loading
def load_page_class(filename, class_name):
    spec = importlib.util.spec_from_file_location("page_module", 
                                                  os.path.join(app_pages_dir, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Load all page classes
ProjectSummaryPage = load_page_class("1_00_📘_Project_Summary.py", "ProjectSummaryPage")
EdaPage = load_page_class("1_01_🔎_EDA.py", "EdaPage")
TrainModelPage = load_page_class("1_02_🧠_Train_Model.py", "TrainModelPage")
PredictPage = load_page_class("1_03_📈_Predict.py", "PredictPage")
TraceabilityPage = load_page_class("1_04_🧪_Traceability.py", "TraceabilityPage")

st.set_page_config(
    page_title="SKU Demand Projection",
    page_icon="🚀",
    layout="wide",
)

PAGES = {
    "📘 Project Summary": ProjectSummaryPage.render,
    "🔎 EDA": EdaPage.render,
    "🧠 Train Model": TrainModelPage.render,
    "📈 Predict": PredictPage.render,
    "🧪 Traceability": TraceabilityPage.render,
}

with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Go to", list(PAGES.keys()))

PAGES[choice]()