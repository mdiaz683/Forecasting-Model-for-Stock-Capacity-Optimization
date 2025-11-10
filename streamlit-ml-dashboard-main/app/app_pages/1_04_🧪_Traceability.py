import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

class TraceabilityPage:
    @staticmethod
    def render():
        st.title("🧪 Traceability")
        
        # Check if model exists
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(model_dir, 'latest.joblib')
        
        if not os.path.exists(model_path):
            st.error("🚫 No trained model found!")
            st.info("Please train a model first in the '🧠 Train Model' section.")
            
            # Show example metrics
            st.subheader("📊 Example Model Performance Metrics")
            
            # Sample metrics for demonstration
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Accuracy",
                    value="85.2%",
                    delta="2.1%"
                )
            
            with col2:
                st.metric(
                    label="Precision",
                    value="87.4%",
                    delta="1.8%"
                )
                
            with col3:
                st.metric(
                    label="Recall",
                    value="83.1%",
                    delta="0.9%"
                )
                
            with col4:
                st.metric(
                    label="F1-Score",
                    value="85.2%",
                    delta="1.4%"
                )
            
            # Example plots
            st.subheader("📈 Example Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sample confusion matrix
                sample_cm = np.array([[45, 5], [8, 42]])
                fig = px.imshow(sample_cm, 
                               text_auto=True, 
                               aspect="auto",
                               title="Sample Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sample ROC curve
                fpr = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
                tpr = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve (AUC=0.85)'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                       mode='lines', 
                                       line=dict(dash='dash'),
                                       name='Random Classifier'))
                fig.update_layout(
                    title='Sample ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Information about metrics
            with st.expander("📚 About Model Metrics"):
                st.markdown("""
                **Key Performance Metrics:**
                
                - **Accuracy**: Overall correctness of predictions
                - **Precision**: How many positive predictions were actually correct
                - **Recall**: How many actual positives were correctly identified
                - **F1-Score**: Harmonic mean of precision and recall
                - **ROC-AUC**: Area under the receiver operating characteristic curve
                - **Confusion Matrix**: Detailed breakdown of correct/incorrect predictions
                
                **For Regression:**
                - **R² Score**: Proportion of variance explained by the model
                - **RMSE**: Root mean squared error
                - **MAE**: Mean absolute error
                """)
            
            return
        
        # Load model if it exists
        try:
            model_data = joblib.load(model_path)
            st.success("✅ Model loaded successfully!")
            
            # Display model info
            st.subheader("🤖 Model Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Type", model_data['model_type'])
            with col2:
                problem_type = "Classification" if model_data['is_classification'] else "Regression"
                st.metric("Problem Type", problem_type)
            with col3:
                st.metric("Features", len(model_data['features']))
            with col4:
                st.metric("Target", model_data['target'])
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        st.markdown("---")
        
        # Model evaluation section
        st.subheader("📊 Model Evaluation")
        st.info("💡 To see detailed metrics, upload test data and run evaluation below.")
        
        # Upload test data for evaluation
        test_file = st.file_uploader("Upload test CSV for evaluation", type=["csv"])
        
        if test_file is not None:
            try:
                test_df = pd.read_csv(test_file)
                st.success(f"✅ Test data loaded! Shape: {test_df.shape}")
                
                # Check if target column exists
                if model_data['target'] not in test_df.columns:
                    st.error(f"❌ Target column '{model_data['target']}' not found in test data!")
                    st.info(f"Available columns: {list(test_df.columns)}")
                    return
                
                # Check if required features are present
                missing_features = set(model_data['features']) - set(test_df.columns)
                if missing_features:
                    st.error(f"❌ Missing required features: {missing_features}")
                    return
                
                if st.button("🔍 Evaluate Model", type="primary"):
                    with st.spinner("Evaluating model..."):
                        try:
                            # Prepare test data
                            X_test = test_df[model_data['features']].copy()
                            y_test = test_df[model_data['target']].copy()
                            
                            # Handle categorical features
                            from sklearn.preprocessing import LabelEncoder
                            categorical_features = X_test.select_dtypes(include=['object']).columns
                            for col in categorical_features:
                                le = LabelEncoder()
                                X_test[col] = le.fit_transform(X_test[col].astype(str))
                            
                            # Handle categorical target for classification
                            if model_data['is_classification'] and y_test.dtype == 'object':
                                le_target = LabelEncoder()
                                y_test = le_target.fit_transform(y_test.astype(str))
                            
                            # Make predictions
                            if model_data['scaler'] is not None:
                                X_test_scaled = model_data['scaler'].transform(X_test)
                                y_pred = model_data['model'].predict(X_test_scaled)
                            else:
                                y_pred = model_data['model'].predict(X_test)
                            
                            # Calculate and display metrics
                            if model_data['is_classification']:
                                # Classification metrics
                                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted')
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')
                                
                                # Display metrics
                                st.subheader("🎯 Classification Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.3f}")
                                with col2:
                                    st.metric("Precision", f"{precision:.3f}")
                                with col3:
                                    st.metric("Recall", f"{recall:.3f}")
                                with col4:
                                    st.metric("F1-Score", f"{f1:.3f}")
                                
                                # Confusion Matrix
                                st.subheader("🔄 Confusion Matrix")
                                cm = confusion_matrix(y_test, y_pred)
                                
                                fig = px.imshow(cm, 
                                               text_auto=True, 
                                               aspect="auto",
                                               title="Confusion Matrix",
                                               labels=dict(x="Predicted", y="Actual"))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Classification Report
                                from sklearn.metrics import classification_report
                                with st.expander("📋 Detailed Classification Report"):
                                    report = classification_report(y_test, y_pred, output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df)
                                
                                # ROC Curve (for binary classification)
                                if len(np.unique(y_test)) == 2:
                                    st.subheader("📈 ROC Curve")
                                    
                                    if hasattr(model_data['model'], 'predict_proba'):
                                        if model_data['scaler'] is not None:
                                            y_scores = model_data['model'].predict_proba(X_test_scaled)[:, 1]
                                        else:
                                            y_scores = model_data['model'].predict_proba(X_test)[:, 1]
                                        
                                        fpr, tpr, _ = roc_curve(y_test, y_scores)
                                        roc_auc = auc(fpr, tpr)
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=fpr, y=tpr, 
                                            name=f'ROC Curve (AUC = {roc_auc:.3f})'
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=[0, 1], y=[0, 1], 
                                            mode='lines', 
                                            line=dict(dash='dash'),
                                            name='Random Classifier'
                                        ))
                                        fig.update_layout(
                                            title='Receiver Operating Characteristic (ROC) Curve',
                                            xaxis_title='False Positive Rate',
                                            yaxis_title='True Positive Rate'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
                            
                            else:
                                # Regression metrics
                                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                
                                # Display metrics
                                st.subheader("📊 Regression Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("R² Score", f"{r2:.3f}")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.3f}")
                                with col3:
                                    st.metric("MAE", f"{mae:.3f}")
                                with col4:
                                    st.metric("MSE", f"{mse:.3f}")
                                
                                # Prediction vs Actual plot
                                st.subheader("📈 Predicted vs Actual")
                                fig = px.scatter(
                                    x=y_test, y=y_pred,
                                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                    title='Predicted vs Actual Values'
                                )
                                
                                # Add perfect prediction line
                                min_val = min(y_test.min(), y_pred.min())
                                max_val = max(y_test.max(), y_pred.max())
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val], 
                                    y=[min_val, max_val],
                                    mode='lines',
                                    line=dict(dash='dash', color='red'),
                                    name='Perfect Prediction'
                                ))
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Residuals plot
                                residuals = y_test - y_pred
                                fig_residuals = px.scatter(
                                    x=y_pred, y=residuals,
                                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                    title='Residuals Plot'
                                )
                                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                                st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"Error loading test data: {str(e)}")
        
        else:
            st.info("📁 Upload test data to see detailed performance metrics.")
            
            # Show model features
            with st.expander("ℹ️ Required Test Data Format"):
                st.markdown(f"""
                **Your test CSV must contain:**
                
                **Target column:** `{model_data['target']}`
                
                **Feature columns:** {', '.join(model_data['features'])}
                
                **Example format:**
                ```csv
                {','.join(model_data['features'])},{model_data['target']}
                1.0,2.5,Category A,0
                2.1,3.2,Category B,1
                ```
                """)
        
        # Feature importance (if available)
        if hasattr(model_data['model'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': model_data['features'],
                'Importance': model_data['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)