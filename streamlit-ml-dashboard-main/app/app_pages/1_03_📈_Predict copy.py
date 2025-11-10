import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import timedelta
from datetime import datetime


from modules.features import get_full_data, get_future_exog, get_split_col, get_df_input
from modules.plotting import save_predictions_history, load_predictions_history, get_relevant_past_predictions, clear_predictions_history, create_forecast_plot_with_history

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

            ### Get data from model

            # Architecture
            forecaster = model_data['forecaster']
            best_params = model_data['best_params']
            best_lags = model_data['best_lags']
            model_type = model_data['model_type']
            test_size = model_data['test_size']

            # Model (v1/v2)
            series_full = model_data['series_full']
            exog_full = model_data['exog_full']
            feature_cols = model_data['features']
            target_col = model_data['target']
            is_ADD = model_data['is_ADD']

            stats = model_data['stats_train']
            metrics = model_data.get('metrics', {})
            scaled_mae = metrics.get('scaled_mae')
            mae       = metrics.get('mae')
            rmse      = metrics.get('rmse')

            
            
            with st.expander("ℹ️ Model Information"):
                st.markdown(f"""
                **Current Model Details:**
                - **Model Type:** {model_type}
                - **Number of features:** {len(feature_cols)}
                - **Featres:** {feature_cols}
                - **Target:** {model_data['target']}

                - **Best Hyperparameters:** {best_params}
                - **Best Lags:** {best_lags}""")

            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        st.markdown("---")
        
        st.subheader("🎯 Choose Prediction Horizon")
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Number of horizon steps:", 1, 20, 4)                                    
        with col2:
            last_date = series_full['timestamp'].max()
            future_date = last_date + timedelta(weeks=horizon)
            st.metric("Predictions Horizon",
                      f"from {last_date.strftime('%Y-%m-%d')} to {future_date.strftime('%Y-%m-%d')}")
         
                    
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                try:
                    if is_ADD == False:
                        target_col = 's_split'
                    
                    series_full_dict, exog_full_dict = get_full_data(series_full, 
                                                            exog_full, 
                                                            target_col)
                    forecaster.fit(series=series_full_dict, exog=exog_full_dict)
                    future_exog_dict = get_future_exog(horizon, series_full)

                    predictions = forecaster.predict(steps=horizon, exog=future_exog_dict)
                    predictions = predictions.rename(columns={"level": 'series_id'})
                    predictions = predictions.copy()
                    predictions.index = pd.to_datetime(predictions.index)
                    predictions.index.name = "timestamp"
                    predictions = predictions.reset_index()
                    print(f"predictions : {predictions}")


                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                
                try:
                    if is_ADD:
                        series_info = series_full[['series_id', 'Brand', 'Resource ID']].drop_duplicates()
                        predictions = predictions.merge(series_info, on='series_id', how='left')

                        predictions = get_split_col(predictions, demand_col="pred")
                        # CORRECCIÓN: predictions ya tiene Brand y Resource ID después del merge
                        predictions = predictions.rename(columns={"split": 'pred'})
                        predictions_for = predictions[['timestamp', 'series_id', 'Brand', 'Resource ID', 'pred']]
                        predictions = predictions[['timestamp', 'series_id', 'pred']]


                    else:
                        predictions[['Brand', 'Resource ID']] = predictions['series_id'].str.split(r'\|\|', expand=True)
                        predictions['p_raw'] = 1.0 / (1.0 + np.exp(-predictions['pred'].astype(float)))
                        den = predictions.groupby(['timestamp', 'Brand'])['p_raw'].transform('sum')
                        predictions['pred'] = np.where(den > 0, predictions['p_raw'] / den, 0.0)
                        predictions = predictions[['timestamp', 'series_id', 'pred']]

                        predictions_for = predictions.copy()
                        predictions_for[['Brand', 'Resource ID']] = predictions_for['series_id'].str.split(r'\|\|', expand=True)
                        # El orden de columnas se puede ajustar directamente
                        predictions_for = predictions_for[['timestamp', 'series_id', 'Brand', 'Resource ID', 'pred']]

                    st.session_state['predictions'] = predictions
                    st.session_state['predictions_for'] = predictions_for
                    
                    # NUEVO: Guardar estas predicciones en el historial
                    # Usar la fecha actual o la última fecha de series_full como fecha de predicción
                    prediction_date = series_full['timestamp'].max()
                    save_predictions_history(predictions, prediction_date, model_dir)
                    st.success(f"✅ Predictions saved to history (date: {prediction_date})")

                
                except Exception as e:
                    st.error(f"Error formating predictions: {str(e)}")
                    
        # MOSTRAR RESULTADOS SI EXISTEN EN SESSION STATE
        if 'predictions_for' in st.session_state:
            predictions = st.session_state['predictions']
            predictions_for = st.session_state['predictions_for']
            
            # Display results
            st.subheader("🎯 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Prediction", f"{np.mean(predictions['pred']):.3f}")
            with col2:
                st.metric("Min Prediction", f"{np.min(predictions['pred']):.3f}")
            with col3:
                st.metric("Max Prediction", f"{np.max(predictions['pred']):.3f}")
            predictions['%'] = predictions['pred']*100
            predictions_for['%'] = predictions_for['pred']*100
            st.dataframe(predictions)
            
            # Filters section
            st.subheader("🔍 Filter Results")
            
            # Inicializar variables de filtro previas si no existen
            if 'prev_brand_filter' not in st.session_state:
                st.session_state.prev_brand_filter = 'All'
            if 'prev_resource_filter' not in st.session_state:
                st.session_state.prev_resource_filter = 'All'
            
            # Create columns for filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Timestamp filter (date range)
                if 'timestamp' in predictions_for.columns:
                    min_date = predictions_for['timestamp'].min()
                    max_date = predictions_for['timestamp'].max()
            
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="date_filter"
                    )
            
            # Determinar qué filtro cambió
            brand_changed = False
            resource_changed = False
            
            if 'brand_filter' in st.session_state:
                brand_changed = st.session_state.brand_filter != st.session_state.prev_brand_filter
            if 'resource_filter' in st.session_state:
                resource_changed = st.session_state.resource_filter != st.session_state.prev_resource_filter
            
            with col2:
                # Brand filter with dynamic options based on Resource ID
                if 'Brand' in predictions_for.columns:
                    # Si cambió el Resource ID, filtrar las brands disponibles
                    if resource_changed and st.session_state.resource_filter != 'All':
                        available_brands = predictions_for[
                            predictions_for['Resource ID'] == st.session_state.resource_filter
                        ]['Brand'].unique().tolist()
                        brands = ['All'] + sorted(available_brands)
                        # Si la brand actual no está disponible, resetear a 'All'
                        default_brand = 'All' if st.session_state.brand_filter not in brands else st.session_state.brand_filter
                    elif not brand_changed and st.session_state.prev_resource_filter != 'All' and st.session_state.resource_filter != 'All':
                        # Mantener el filtro de brands basado en resource
                        available_brands = predictions_for[
                            predictions_for['Resource ID'] == st.session_state.resource_filter
                        ]['Brand'].unique().tolist()
                        brands = ['All'] + sorted(available_brands)
                        default_brand = st.session_state.get('brand_filter', 'All')
                    else:
                        brands = ['All'] + sorted(predictions_for['Brand'].unique().tolist())
                        default_brand = st.session_state.get('brand_filter', 'All')
                    
                    # Encontrar el índice de la opción por defecto
                    default_index = brands.index(default_brand) if default_brand in brands else 0
                    
                    selected_brand = st.selectbox(
                        "Select Brand:",
                        options=brands,
                        index=default_index,
                        key="brand_filter"
                    )
            
            with col3:
                # ResourceID filter with dynamic options based on Brand
                if 'Resource ID' in predictions_for.columns:
                    # Si cambió la Brand, filtrar los Resource IDs disponibles
                    if brand_changed and st.session_state.brand_filter != 'All':
                        available_resources = predictions_for[
                            predictions_for['Brand'] == st.session_state.brand_filter
                        ]['Resource ID'].unique().tolist()
                        resources = ['All'] + sorted(available_resources)
                        # Si el resource actual no está disponible, resetear a 'All'
                        default_resource = 'All' if st.session_state.resource_filter not in resources else st.session_state.resource_filter
                    elif not resource_changed and st.session_state.prev_brand_filter != 'All' and st.session_state.brand_filter != 'All':
                        # Mantener el filtro de resources basado en brand
                        available_resources = predictions_for[
                            predictions_for['Brand'] == st.session_state.brand_filter
                        ]['Resource ID'].unique().tolist()
                        resources = ['All'] + sorted(available_resources)
                        default_resource = st.session_state.get('resource_filter', 'All')
                    else:
                        resources = ['All'] + sorted(predictions_for['Resource ID'].unique().tolist())
                        default_resource = st.session_state.get('resource_filter', 'All')
                    
                    # Encontrar el índice de la opción por defecto
                    default_index = resources.index(default_resource) if default_resource in resources else 0
                    
                    selected_resource = st.selectbox(
                        "Select Resource ID:",
                        options=resources,
                        index=default_index,
                        key="resource_filter"
                    )
            
            # Actualizar los valores previos
            st.session_state.prev_brand_filter = selected_brand
            st.session_state.prev_resource_filter = selected_resource
            
            # Apply filters
            filtered_df = predictions_for.copy()
            
            # Filter by timestamp
            if 'timestamp' in predictions_for.columns and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= start_date) & 
                    (filtered_df['timestamp'].dt.date <= end_date)
                ]
            
            # Filter by Brand
            if 'Brand' in predictions_for.columns and selected_brand != 'All':
                filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
            
            # Filter by ResourceID
            if 'Resource ID' in predictions_for.columns and selected_resource != 'All':
                filtered_df = filtered_df[filtered_df['Resource ID'] == selected_resource]
            
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} of {len(predictions_for)} records")
            st.dataframe(filtered_df)

            filename = f'latest_split_preds_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            model_path2 = os.path.join(model_dir, filename)
            
            filtered_df.to_excel(model_path2, index=False)

            # Mostrar mensaje en la app
            st.success(f"💾 Model saved to: {model_path2}")

            # Leer el archivo como binario para descargarlo
            with open(model_path2, 'rb') as f:
                st.download_button(
                    label="📥 Download split predictions in Excel",
                    data=f,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
'''
            st.markdown("---")
            st.subheader("📊 Forecast Visualization")

            # Cargar historial de predicciones
            predictions_history = load_predictions_history(model_dir)

            # Información del historial
            if predictions_history is not None:
                n_predictions = len(predictions_history)
                first_pred = predictions_history['prediction_date'].min()
                last_pred = predictions_history['prediction_date'].max()
                st.info(f"📚 Predictions History: {n_predictions} records from {first_pred} to {last_pred}")
            else:
                st.warning("⚠️ No predictions history found. Start making predictions to build history.")

            # Selector de serie para visualizar
            available_series = filtered_df['series_id'].unique().tolist()

            if len(available_series) > 0:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_viz_series = st.selectbox(
                        "Select Series to Visualize:",
                        options=available_series,
                        key="viz_series_selector"
                    )
                
                with col2:
                    # Botón para limpiar historial (opcional)
                    if st.button("🗑️ Clear History", help="Delete all prediction history"):
                        clear_predictions_history(model_dir)
                        st.rerun()
                
                # Crear y mostrar la gráfica
                fig = create_forecast_plot_with_history(
                    series_full=series_full,
                    predictions_for=predictions,
                    predictions_history=predictions_history,
                    selected_series_id=selected_viz_series,
                    model_start_date='2025-07-09',
                    today='2025-07-09'  # En producción: datetime.now()
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization with current filters.")
'''
                                 
                    
            