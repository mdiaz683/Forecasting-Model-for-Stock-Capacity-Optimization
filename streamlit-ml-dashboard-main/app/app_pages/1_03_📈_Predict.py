import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PredictPage:
    @staticmethod
    def render():
        st.title("📈 Predict")
        
        # Check if model exists
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(model_dir, 'latest.joblib')
        
        if not os.path.exists(model_path):
            st.error("🚫 No trained model found!")
            st.info("Please train a model first in the '🧠 Train Model' section.")
            return
        
        # Load model
        try:
            model_data = joblib.load(model_path)
            st.success("✅ Model loaded successfully!")
            
            # Display model info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_data['model_type'])
            with col2:
                st.metric("Features", len(model_data['features']))
            with col3:
                problem_type = "Classification" if model_data['is_classification'] else "Regression"
                st.metric("Problem Type", problem_type)
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        st.markdown("---")
        
        # Prediction options
        st.subheader("🎯 Choose Prediction Method")
        prediction_mode = st.radio(
            "Select prediction mode:",
            ["Single Prediction", "Batch Prediction (CSV)"]
        )
        
        if prediction_mode == "Single Prediction":
            st.subheader("📝 Enter Feature Values")
            
            # Create input fields for each feature
            input_data = {}
            
            col1, col2 = st.columns(2)
            for i, feature in enumerate(model_data['features']):
                with col1 if i % 2 == 0 else col2:
                    # For simplicity, assume all inputs are numeric
                    # In a real app, you'd want to store feature types
                    input_data[feature] = st.number_input(
                        f"Enter {feature}:",
                        value=0.0,
                        key=f"input_{feature}"
                    )
            
            # Make prediction
            if st.button("🔮 Make Prediction", type="primary"):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Handle categorical features (basic approach)
                    categorical_features = input_df.select_dtypes(include=['object']).columns
                    for col in categorical_features:
                        le = LabelEncoder()
                        input_df[col] = le.fit_transform(input_df[col].astype(str))
                    
                    # Make prediction
                    if model_data['scaler'] is not None:
                        # Use scaler for linear models
                        input_scaled = model_data['scaler'].transform(input_df)
                        prediction = model_data['model'].predict(input_scaled)
                    else:
                        # Direct prediction for tree-based models
                        prediction = model_data['model'].predict(input_df)
                    
                    # Display result
                    st.subheader("🎯 Prediction Result")
                    if model_data['is_classification']:
                        if hasattr(model_data['model'], 'predict_proba'):
                            probabilities = model_data['model'].predict_proba(
                                model_data['scaler'].transform(input_df) 
                                if model_data['scaler'] is not None 
                                else input_df
                            )
                            st.success(f"**Predicted Class:** {prediction[0]}")
                            
                            # Show probabilities
                            if len(probabilities[0]) > 1:
                                st.subheader("🎲 Class Probabilities")
                                prob_df = pd.DataFrame({
                                    'Class': range(len(probabilities[0])),
                                    'Probability': probabilities[0]
                                })
                                st.bar_chart(prob_df.set_index('Class'))
                        else:
                            st.success(f"**Predicted Class:** {prediction[0]}")
                    else:
                        st.success(f"**Predicted Value:** {prediction[0]:.3f}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        else:  # Batch Prediction
            st.subheader("📁 Upload CSV for Batch Prediction")
            
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Load data
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Data loaded! Shape: {df.shape}")
                    
                    # Check if required features are present
                    missing_features = set(model_data['features']) - set(df.columns)
                    if missing_features:
                        st.error(f"❌ Missing required features: {missing_features}")
                        st.info("Required features: " + ", ".join(model_data['features']))
                        return
                    
                    # Show data preview
                    st.subheader("📊 Data Preview")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Generate Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            try:
                                # Prepare data
                                X = df[model_data['features']].copy()
                                
                                # Handle categorical features
                                categorical_features = X.select_dtypes(include=['object']).columns
                                for col in categorical_features:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                                
                                # Make predictions
                                if model_data['scaler'] is not None:
                                    X_scaled = model_data['scaler'].transform(X)
                                    predictions = model_data['model'].predict(X_scaled)
                                else:
                                    predictions = model_data['model'].predict(X)
                                
                                # Add predictions to dataframe
                                results_df = df.copy()
                                results_df['Prediction'] = predictions
                                
                                # Add probabilities for classification
                                if model_data['is_classification'] and hasattr(model_data['model'], 'predict_proba'):
                                    probabilities = model_data['model'].predict_proba(
                                        X_scaled if model_data['scaler'] is not None else X
                                    )
                                    
                                    # Add probability columns
                                    for i in range(probabilities.shape[1]):
                                        results_df[f'Probability_Class_{i}'] = probabilities[:, i]
                                
                                # Display results
                                st.subheader("🎯 Prediction Results")
                                st.dataframe(results_df)
                                
                                # Summary statistics
                                st.subheader("📊 Prediction Summary")
                                if model_data['is_classification']:
                                    pred_counts = pd.Series(predictions).value_counts()
                                    st.bar_chart(pred_counts)
                                else:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Mean Prediction", f"{np.mean(predictions):.3f}")
                                    with col2:
                                        st.metric("Min Prediction", f"{np.min(predictions):.3f}")
                                    with col3:
                                        st.metric("Max Prediction", f"{np.max(predictions):.3f}")
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name='predictions.csv',
                                    mime='text/csv'
                                )
                                
                            except Exception as e:
                                st.error(f"Error making predictions: {str(e)}")
                                st.exception(e)
                
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            
            else:
                st.info("📁 Please upload a CSV file for batch prediction.")
                
                # Show required features
                with st.expander("ℹ️ Required CSV Format"):
                    st.markdown(f"""
                    **Your CSV must contain these columns:**
                    
                    {', '.join(model_data['features'])}
                    
                    **Example:**
                    ```csv
                    {','.join(model_data['features'])}
                    1.0,2.5,Category A
                    2.1,3.2,Category B
                    ```
                    """)
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            st.markdown(f"""
            **Current Model Details:**
            - **Model Type:** {model_data['model_type']}
            - **Problem Type:** {'Classification' if model_data['is_classification'] else 'Regression'}
            - **Features:** {len(model_data['features'])}
            - **Target:** {model_data['target']}
            
            **Required Features:** {', '.join(model_data['features'])}
            """)

                