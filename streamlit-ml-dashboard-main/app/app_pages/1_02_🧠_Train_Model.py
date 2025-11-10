import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px



class TrainModelPage:
    @staticmethod
    def render():
        st.title("🧠 Train Model")
        
        # File uploader
        data_file = st.file_uploader("Upload Training CSV", type=["csv"])
        
        if data_file:
            try:
                df = pd.read_csv(data_file)
                st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                # Target selection
                st.subheader("🎯 Target Selection")
                target_col = st.selectbox("Select target column:", df.columns.tolist())
                
                if target_col:
                    # Determine problem type
                    is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 10
                    problem_type = "Classification" if is_classification else "Regression"
                    
                    st.info(f"Detected problem type: **{problem_type}**")
                    
                    # Feature selection
                    st.subheader("🔧 Feature Selection")
                    feature_cols = [col for col in df.columns if col != target_col]
                    selected_features = st.multiselect(
                        "Select features:", 
                        feature_cols, 
                        default=feature_cols[:10]  # Default to first 10 features
                    )
                    
                    if selected_features:
                        # Data preprocessing info
                        st.subheader("📊 Data Overview")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Samples", len(df))
                        with col2:
                            st.metric("Features", len(selected_features))
                        with col3:
                            if is_classification:
                                st.metric("Classes", df[target_col].nunique())
                            else:
                                st.metric("Target Range", f"{df[target_col].min():.2f} - {df[target_col].max():.2f}")
                        
                        # Model selection
                        st.subheader("🤖 Model Selection")
                        if is_classification:
                            model_type = st.selectbox(
                                "Choose classifier:",
                                ["Logistic Regression", "Random Forest"]
                            )
                        else:
                            model_type = st.selectbox(
                                "Choose regressor:",
                                ["Linear Regression", "Random Forest"]
                            )
                        
                        # Training parameters
                        st.subheader("⚙️ Training Parameters")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            test_size = st.slider("Test size (%):", 10, 40, 20) / 100
                            random_state = st.number_input("Random state:", value=42, min_value=0)
                        
                        with col2:
                            if "Random Forest" in model_type:
                                n_estimators = st.slider("Number of trees:", 10, 200, 100)
                                max_depth = st.slider("Max depth:", 1, 20, 10)
                        
                        # Train button
                        if st.button("🚀 Train Model", type="primary"):
                            with st.spinner("Training model..."):
                                # Prepare data
                                X = df[selected_features].copy()
                                y = df[target_col].copy()
                                
                                # Handle categorical features
                                categorical_features = X.select_dtypes(include=['object']).columns
                                for col in categorical_features:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                                
                                # Handle categorical target for classification
                                if is_classification and y.dtype == 'object':
                                    le_target = LabelEncoder()
                                    y = le_target.fit_transform(y.astype(str))
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=random_state
                                )
                                
                                # Scale features
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Select and train model
                                if model_type == "Logistic Regression":
                                    model = LogisticRegression(random_state=random_state)
                                elif model_type == "Linear Regression":
                                    model = LinearRegression()
                                elif "Random Forest" in model_type:
                                    if is_classification:
                                        model = RandomForestClassifier(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=random_state
                                        )
                                    else:
                                        model = RandomForestRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=random_state
                                        )
                                
                                # Train model
                                if "Random Forest" in model_type:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                else:
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)
                                
                                # Calculate metrics
                                if is_classification:
                                    accuracy = accuracy_score(y_test, y_pred)
                                    st.success(f"✅ Model trained successfully!")
                                    st.metric("Accuracy", f"{accuracy:.3f}")
                                    
                                    # Classification report
                                    with st.expander("📊 Detailed Classification Report"):
                                        report = classification_report(y_test, y_pred, output_dict=True)
                                        st.json(report)
                                else:
                                    mse = mean_squared_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    st.success(f"✅ Model trained successfully!")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("R² Score", f"{r2:.3f}")
                                    with col2:
                                        st.metric("RMSE", f"{mse**0.5:.3f}")
                                    
                                    # Prediction vs Actual plot
                                    fig = px.scatter(
                                        x=y_test, y=y_pred,
                                        labels={'x': 'Actual', 'y': 'Predicted'},
                                        title='Predicted vs Actual Values'
                                    )
                                    fig.add_shape(
                                        type="line", line=dict(dash="dash"),
                                        x0=y_test.min(), y0=y_test.min(),
                                        x1=y_test.max(), y1=y_test.max()
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Save model
                                model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                                os.makedirs(model_dir, exist_ok=True)
                                
                                model_data = {
                                    'model': model,
                                    'scaler': scaler if "Random Forest" not in model_type else None,
                                    'features': selected_features,
                                    'target': target_col,
                                    'model_type': model_type,
                                    'is_classification': is_classification
                                }
                                
                                model_path = os.path.join(model_dir, 'latest.joblib')
                                joblib.dump(model_data, model_path)
                                
                                st.success(f"💾 Model saved to: {model_path}")
                        
                    else:
                        st.warning("Please select at least one feature.")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)
        
        else:
            st.info("Please upload a CSV file to start training.")
            
            # Instructions
            with st.expander("📋 Training Instructions"):
                st.markdown("""
                **Steps to train your model:**
                
                1. **Upload CSV**: Your data should have headers
                2. **Select Target**: Choose the column you want to predict
                3. **Select Features**: Choose input columns for prediction
                4. **Choose Model**: Pick the appropriate algorithm
                5. **Set Parameters**: Adjust training settings
                6. **Train**: Click the train button
                
                **Supported formats:**
                - CSV files with headers
                - Numeric and categorical features
                - Classification and regression problems
                """)