import streamlit as st

class ProjectSummaryPage:
    @staticmethod
    def render():
        st.title("📘 Project Summary")
        
        # Hero section
        st.markdown("""
        ## 🚀 Predictive Analytics Dashboard
        
        A comprehensive machine learning platform for data exploration, model training, 
        and predictive analytics built with Streamlit.
        """)
        
        # Key features
        st.subheader("✨ Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔎 Data Exploration**
            - Interactive data upload and preview
            - Statistical summaries and visualizations
            - Missing value analysis
            - Correlation matrices
            - Distribution plots
            
            **🧠 Model Training**
            - Support for ADD forecasting
            - LightGBM model
            - Automated data preprocessing
            - Model persistence and saving
            - Multiple graphs for visualize performance
            """)
        
        with col2:
            st.markdown("""
            **📈 Predictions**
            - Predictions from CSV files
            - Downloadable prediction results
            - Dynamic table with filters to visualize results
            """)

        # How to use
        st.subheader("📖 How to Use This App")
        
        steps = [
            ("1️⃣ **Explore Data**", "Upload your CSV file in the EDA section to understand your data"),
            ("2️⃣ **Train Model**", "Use the Train Model section to build and train your ML model"),
            ("3️⃣ **Make Predictions**", "Generate predictions for new data using the Predict section"),
            ("4️⃣ **Evaluate Performance**", "Review model metrics and performance in the Model Metrics section")
        ]
        
        for step, description in steps:
            st.markdown(f"**{step}**")
            st.markdown(f"   {description}")
            st.markdown("")
