import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt

from modules.features import load_data, series_exog, train_test_split, calculate_stats_train, normalize_series, create_time_series_dicts, get_split_col, get_df_input, get_series_preds, get_final_df

from modules.plotting import plot_series, plot_train_test_split, plot_results, plot_results_dual_scale, create_residuals_plot, create_comparison_plot

from modules.lgbm_architecture import create_forecaster, tune_forecaster, create_final_forecaster, backtest_forecaster, fit_predict

from modules.metrics import format_results, residuals, get_full_metrics


targets = ['ADD', 'split']

class TrainModelPage:
    @staticmethod
    def render():
        st.title("🧠 Train Model")
        
        # File uploader
        data_file = st.file_uploader("Upload Training CSV", type=["csv"])
        
        if data_file:
            try:
                features = pd.read_csv(data_file)
                features['timestamp'] = pd.to_datetime(features['timestamp']).dt.date

                st.success(f"Data loaded successfully! Shape: {features.shape}")
                
                # Target selection
                st.subheader("🎯 Target Selection")
                target_col = st.selectbox("Select target column:", targets)
                
                if target_col:
                    # Determine problem type
                    is_ADD = (target_col == 'ADD')
                    problem_type = "Split projection based on historic ADD" if is_ADD else "Split projection based on historic Split"
                    
                    st.info(f"Detected problem type: **{problem_type}**")
                    
                    try:
                        series_ini, exog_ini = series_exog(features)
                        df_full = get_split_col(series_ini)
                        if is_ADD:
                            series, exog = series_exog(features)
                            feature_cols = [col for col in series.columns]
                            # exog_cols = [col for col in exog.columns]
                        else:
                            feature_cols = [col for col in df_full.columns]
                            # exog_cols = [col for col in exog_ini.columns]
                            df_input = get_df_input(df_full)
                            wk_feats = (exog_ini[['timestamp', 'sin_week', 'cos_week', 'week_number']]
                                        .drop_duplicates(subset=['timestamp']))
                            df_input = df_input.merge(wk_feats, on='timestamp', how='left')
                            series, exog = series_exog(df_input)
                    except Exception as e:
                        st.error(f"Error processing series and exogenous variables: {str(e)}")
                        st.exception(e)
                        return


                    st.subheader("📊 Data Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Samples", len(series))
                    with col2:
                        st.metric("Features", len(feature_cols))
                    with col3:
                        # Verificar que el target existe en series
                        if target_col in series.columns:
                            st.metric("Target Range", 
                                     f"{series[target_col].min():.2f} - {series[target_col].max():.2f}")
                        else:
                            st.metric("Target Range", "N/A")
                    
                    st.subheader("Sreies Data Preview")
                    if is_ADD:
                        st.dataframe(series.head(5))
                    else:
                        df_full = df_full[['timestamp', 'series_id'] + [col for col in feature_cols if col not in ['timestamp', 'series_id']]]
                        st.dataframe(df_full.head(5))
                    
                    st.subheader("Exogenous Features Preview")
                    st.dataframe(exog.head(5))
                    
                    # Model selection
                    st.subheader("🤖 Model Selection")
                    model_type = st.selectbox(
                        "Choose classifier:",
                        ["LightGBM"]
                    )
                    
                    # Training parameters
                    st.subheader("⚙️ Test Size")
                    col1, col2 = st.columns(2)
                    with col1:
                        test_size = st.slider("Number of weeks for test set:", 1, 20, 4)
                        try:
                            series_train, series_test, exog_train, exog_test = train_test_split(
                                series, exog, test_weeks=test_size
                            )
                        except Exception as e:
                            st.error(f"Error splitting data: {str(e)}")
                            return
                                                
                    with col2:
                        st.metric("Train range",
                                  f"{series_train['timestamp'].min().strftime('%Y-%m-%d')} - {series_train['timestamp'].max().strftime('%Y-%m-%d')}")

                        st.metric("Test range",
                                  f"{series_test['timestamp'].min().strftime('%Y-%m-%d')} - {series_test['timestamp'].max().strftime('%Y-%m-%d')}")

                                        
                    # Train button
                    if st.button("🚀 Train Model", type="primary"):
                        with st.spinner("Training model..."):
                            try:
                                # Preparación de datos según tipo
                                if is_ADD:
                                    series_train, series_test = normalize_series(series_train, series_test)
                                    series_dict_train, series_dict_test, exog_dict_train, exog_dict_test = create_time_series_dicts(
                                        series_train, series_test, 
                                        exog_train, exog_test
                                    )
                                    stats_train = calculate_stats_train(series_train)
                                else:
                                    series_dict_train, series_dict_test, exog_dict_train, exog_dict_test = create_time_series_dicts(
                                        series_train, series_test, 
                                        exog_train, exog_test, 
                                        target_col='s_split'
                                    )
                                    stats_train = None  # No se usa para split
                                
                                # Creación y entrenamiento del forecaster
                                '''
                                best_params, best_lags = tune_forecaster(series_dict_train, exog_dict_train)
                                '''
                                # Manual Selection (TEMPORARY)
                                if is_ADD:
                                    best_lags= [1, 2, 3, 4]
                                    best_params = {'n_estimators': 200, 'max_depth': 5, 'num_leaves': 47, 'min_child_samples': 90, 'learning_rate': 0.06107135917674153, 'feature_fraction': 0.7, 'bagging_fraction': 1.0, 'lambda_l1': 0.7000000000000001, 'lambda_l2': 0.30000000000000004}
                                else:
                                    best_lags= [1, 2, 3, 4]
                                    best_params = {'n_estimators': 300, 'max_depth': 7, 'num_leaves': 47, 'min_child_samples': 160, 'learning_rate': 0.04801856153426353, 'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'lambda_l1': 0.7000000000000001, 'lambda_l2': 0.5}
                                
                                final_forecaster = create_final_forecaster(best_params, best_lags)
                                
                                # Predicciones
                                predictions = fit_predict(
                                    final_forecaster, 
                                    series_dict_train, 
                                    series_dict_test, 
                                    exog_dict_train=exog_dict_train, 
                                    exog_dict_test=exog_dict_test,
                                    steps=test_size
                                )
                                
                                # Formateo de resultados
                                if is_ADD:
                                    ADD_predictions, ADD_scale_mae_global, ADD_mae_global = format_results(
                                        predictions,
                                        series_dict_test,
                                        stats_train,
                                        'v1') # df with ADD as target
                                    
                                    f_predictions = get_final_df(ADD_predictions, series_test) # df with split as target
                                    mae_global = f_predictions["mae"].mean()
                                    print(f"f_predictions: {f_predictions}")

                                    ADD_y_pred = ADD_predictions['pred']
                                    ADD_y_test = ADD_predictions['y']
                                    ADD_rmse = mean_squared_error(ADD_y_test, ADD_y_pred) ** 0.5

                                
                                else:
                                    f_predictions, scale_mae_global, mae_global = format_results(
                                        predictions,
                                        series_dict_test,
                                        df_full,
                                        'v2'
                                    )
                                print(f"f_predictions : {f_predictions}")
                                st.subheader("Predictions Overview")
                                st.dataframe(f_predictions)

                                y_pred = f_predictions['pred'] #split
                                y_test = f_predictions['y']

                                mse = mean_squared_error(y_test, y_pred)
                                rmse = mse ** 0.5

                                st.session_state.model_results = {
                                    'y_test': y_test,
                                    'y_pred': y_pred,
                                    'f_predictions': f_predictions,
                                    'series': series,
                                    'df_full': df_full,
                                    'mae_global': mae_global,              # ✅ AGREGADO
                                    'rmse': rmse,                          # ✅ AGREGADO
                                    'model_trained': True,
                                    'ADD_predictions': ADD_predictions if is_ADD else None,
                                    'ADD_y_pred': ADD_y_pred if is_ADD else None,
                                    'ADD_y_test': ADD_y_test if is_ADD else None,
                                    'ADD_rmse': ADD_rmse if is_ADD else None,
                                    'ADD_scaled_mae': ADD_scale_mae_global if is_ADD else None,
                                    'ADD_mae': ADD_mae_global if is_ADD else None,                                    
                                }

                                st.success("✅ Model trained successfully! Results saved.")
                                                                
                                ### Save model for PREDICTIONS
                                model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                                os.makedirs(model_dir, exist_ok=True)
                                
                                model_data = {
                                    'series_full': series,
                                    'exog_full': exog,
                                    'forecaster': final_forecaster,
                                    'best_params': best_params,
                                    'best_lags': best_lags,
                                    'features': feature_cols,
                                    'target': target_col,
                                    'model_type': model_type,
                                    'is_ADD': is_ADD,
                                    'test_size': test_size,
                                    'stats_train': stats_train if is_ADD else None,
                                    'metrics': {
                                        'mae': mae_global,
                                        'rmse': rmse,
                                        'ADD_scaled_mae': ADD_scale_mae_global if is_ADD else None,
                                        'ADD_mae': ADD_mae_global if is_ADD else None,
                                        'ADD_rmse': ADD_rmse if is_ADD else None,                                        
                                    }
                                }
                                
                                model_filename = 'latest.joblib'
                                model_path = os.path.join(model_dir, model_filename)
                                joblib.dump(model_data, model_path)
                                
                                st.success(f"💾 Model saved to: {model_path}")
                                
                                # Botón de descarga
                                with open(model_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Model",
                                        data=f,
                                        file_name=model_filename,
                                        mime="application/octet-stream"
                                    )
                                    
                            except Exception as e:
                                st.error(f"Error during training: {str(e)}")
                                st.exception(e)

                    if 'model_results' in st.session_state:
                        # Cargar variables base
                        y_test = st.session_state.model_results['y_test']
                        y_pred = st.session_state.model_results['y_pred']
                        f_predictions = st.session_state.model_results['f_predictions']
                        series = st.session_state.model_results['series']
                        df_full = st.session_state.model_results['df_full']
                        mae_global = st.session_state.model_results['mae_global']
                        rmse = st.session_state.model_results['rmse']

                        # Inicializar col con valor por defecto
                        col = 'split'

                        # Determinar qué métricas y datos usar
                        if is_ADD:
                            # Cargar todas las variables ADD_
                            ADD_y_test = st.session_state.model_results['ADD_y_test']
                            ADD_y_pred = st.session_state.model_results['ADD_y_pred']
                            ADD_rmse = st.session_state.model_results['ADD_rmse']
                            ADD_scale_mae_global = st.session_state.model_results['ADD_scaled_mae']
                            ADD_mae_global = st.session_state.model_results['ADD_mae']
                            ADD_predictions = st.session_state.model_results['ADD_predictions']
                            
                            # Mostrar selector de métricas
                            st.subheader("Select preferred metrics")
                            metrics = st.selectbox("Select metrics type:", ["ADD metrics", "split metrics"])
                            ADD_metrics = (metrics == 'ADD metrics')
                            
                            # Asignar datos según selección
                            if ADD_metrics:
                                # Usar datos ADD_
                                y_test = ADD_y_test
                                y_pred = ADD_y_pred
                                mae_global = ADD_mae_global
                                rmse = ADD_rmse
                                f_predictions = ADD_predictions
                                col = 'ADD'  # ✅ Para gráficos con ADD
                                
                                # Mostrar métricas con Scaled MAE
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Scaled MAE global", f"{ADD_scale_mae_global:.4f}")
                                with col2:
                                    st.metric("MAE global", f"{mae_global:.4f}")
                                with col3:
                                    st.metric("RMSE", f"{rmse:.4f}")
                            else:
                                # Usar datos split (ya asignados al inicio)
                                series = df_full
                                col = 'split'  # ✅ Para gráficos con split
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("MAE global", f"{mae_global:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                

                        else:
                            # is_ADD es False: usar directamente datos split
                            col = 'split'  # ✅ Asegurar que col está definido
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("MAE global", f"{mae_global:.4f}")
                            with col2:
                                st.metric("RMSE", f"{rmse:.4f}")
                        
                        # Generar gráficas (con los datos ya asignados según la selección)
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
                        
                        tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Residuals by Horizon", "Time Series Comparison"])
                        
                        with tab1:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            if is_ADD and ADD_metrics:
                                fig2, resumen = create_residuals_plot(f_predictions, error_col='abs_error_z')
                                st.plotly_chart(fig2, use_container_width=True)
                                st.subheader("Statistical Summary - Normalized Data")

                            else:
                                fig2, resumen = create_residuals_plot(f_predictions, error_col='abs_error')
                                st.plotly_chart(fig2, use_container_width=True)
                                st.subheader("Statistical Summary")
                            st.dataframe(resumen.style.format("{:.6f}"), use_container_width=True)
                        
                        with tab3:
                            st.header("Time Series Comparison: Real vs Predictions")
                            
                            # Selector de serie
                            available_series = f_predictions["series_id"].astype(str).unique()
                            selected_series = st.selectbox(
                                "Select a series to visualize:",
                                options=available_series,
                                index=0
                            )
                            
                            # Selector de escala
                            scale_type = st.radio(
                                "Scale type:",
                                options=["Automatic", "Fixed (0-1)"],
                                horizontal=True
                            )
                            
                            # Debug prints
                            print(f"SELECTED SERIES: {selected_series}")
                            print(f"COL VALUE: {col}")
                            print(f"F_PREDICTIONS series_id sample: {f_predictions['series_id'].head()}")
                            print(f"SERIES 3 series_id sample: {series}")
                            
                            # Crear gráfico
                            fig3, mae = create_comparison_plot(
                                predictions_df=f_predictions,
                                real_serie=series,
                                series_id=selected_series,
                                col=col  # ✅ Ahora col siempre está definido correctamente
                            )
                            
                            if fig3:
                                # Aplicar escala fija si se selecciona
                                if scale_type == "Fixed (0-1)":
                                    fig3.update_yaxes(range=[-0.1, 1.1])
                                    fig3.update_layout(title=fig3.layout.title.text + " (Scale 0-1)")
                                
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                # Mostrar estadísticas de la serie
                                st.subheader("Series Statistics")
                                serie_stats = f_predictions[f_predictions["series_id"].astype(str) == selected_series][
                                    ['y', 'pred', 'abs_error']
                                ].describe()
                                st.dataframe(serie_stats, use_container_width=True)
                            else:
                                st.warning(f"No data available for series {selected_series}")
                                        
                    else:
                        st.warning("Press 'Train Model' button to start training your forecaster.")
                        
                else:
                    st.warning("Please select a target column.")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)
        
        else:
            st.info("Please upload a CSV file to start training.")
            
            # Instructions
            with st.expander("📋 Training Instructions"):
                st.markdown("""
                **Steps to train your forecaster:**
                
                1. **Upload CSV**: Your data should have headers and a 'timestamp' column
                2. **Select Target**: Choose between 'ADD' or 'split'
                3. **Select Features**: Choose input columns for forecasting
                4. **Choose Model**: LightGBM forecaster with skforecast
                5. **Set Test Size**: Number of weeks to use for testing
                6. **Train**: Click the train button
                
                **Supported formats:**
                - CSV files with headers
                - Time series data with timestamp column
                - Numeric features and exogenous variables
                
                **Model Output:**
                - Trained forecaster saved as .joblib file
                - Evaluation metrics (MAE, Scaled MAE, RMSE)
                - Prediction vs Actual visualization
                """)