import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def get_series_to_plot(data, n_plots=4, specific_series=None):
    if specific_series is not None:
        if isinstance(specific_series, str):
            return [specific_series]
        elif isinstance(specific_series, (list, np.ndarray)):
            return list(specific_series)
    
    # Obtener todas las series disponibles
    if isinstance(data, dict):
        all_series = list(data.keys())
    elif isinstance(data, pd.DataFrame):
        all_series = data["series_id"].unique().tolist()
    elif isinstance(data, (list, np.ndarray)):
        all_series = list(data)
    else:
        print("Data type not suported.")
        return []

    if len(all_series) == 0:
        print("No series found in data.")
        return

    series_to_plot = np.random.choice(all_series, size=min(n_plots, len(all_series)), replace=False)
    
    return series_to_plot



def plot_series(data, column='ADD', n_plots=2):
    """Plot random time series from the dataset."""
    series_to_plot = get_series_to_plot(data, n_plots=n_plots)
    
    for sku in series_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
        data.query('series_id == @sku').plot(
            x='timestamp',
            y=[column],
            ax=ax,
            title=sku,
            linewidth=0.3,
            legend=False,
        )
        ax.set_ylabel(column)



def plot_train_test_split(series_dict_train, series_dict_test, n_plots=2, 
                          ylabel='z_ADD', cutoff_date=None, figsize=(10, 3)):
    """Plot train/test split for random series from dictionaries."""
    # Get all available series IDs from both dicts
    data = list(set(series_dict_train.keys()) | set(series_dict_test.keys()))
    series_to_plot = get_series_to_plot(data, n_plots=n_plots)

    for sku in series_to_plot:
        if sku not in series_dict_train and sku not in series_dict_test:
            print(f"SKU {sku} not found in train or test; skipping.")
            continue

        # Build full series by concatenating train and test
        parts = []
        if sku in series_dict_train and len(series_dict_train[sku]) > 0:
            parts.append(series_dict_train[sku])
        if sku in series_dict_test and len(series_dict_test[sku]) > 0:
            parts.append(series_dict_test[sku])
        
        if not parts:
            print(f"SKU {sku} has no data; skipping.")
            continue
            
        series_full = pd.concat(parts).sort_index()

        fig, ax = plt.subplots(figsize=figsize)

        series_full.plot(ax=ax, color='lightgray', label='full (train+test)', 
                        linewidth=1.0, alpha=0.7)

        if sku in series_dict_train and len(series_dict_train[sku]) > 0:
            series_dict_train[sku].plot(ax=ax, color='blue', label='train', linewidth=2)

        if sku in series_dict_test and len(series_dict_test[sku]) > 0:
            series_dict_test[sku].plot(ax=ax, color='orange', label='test', linewidth=2)

        if cutoff_date is not None:
            ax.axvline(x=cutoff_date, color='red', linestyle='--', 
                      linewidth=1.5, label='cutoff', alpha=0.8)

        ax.set_title(f"SKU {sku} — Train/Test Split")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Date")
        ax.legend()
        plt.tight_layout()
        plt.show()



def plot_results(predictions_df, n_val_weeks, real_serie, specific_series=None, n_plots=4, col='ADD'):
    """
    Grafica los resultados del predictions_df para series seleccionadas aleatoriamente.
    """
    figsize=(10, 3)
    series_to_plot = get_series_to_plot(real_serie, n_plots, specific_series)
    
    for series_id in series_to_plot:
        series_id = str(series_id)

        ### SUMMARY
        print(f"Serie: {series_id}")
        df_serie = (predictions_df.loc[predictions_df["series_id"].astype(str) == series_id,
                         ['y', "pred", "abs_error"]]
                    .sort_index())
        print(df_serie)
        print(f"MAE {df_serie["abs_error"].mean()}")

        fig, ax = plt.subplots(figsize=figsize)    
        
        try:
            # Serie real (train)
            serie_real = real_serie[real_serie["series_id"] == series_id].copy()
            serie_real = serie_real.set_index("timestamp")[col].sort_index()
            
            if serie_real.empty:
                print(f"Warning: No real data for series_id {series_id}")
                plt.close(fig)
                continue
            
            serie_real.plot(ax=ax, color="black", label="Real (train)")
            
            # Predicciones para la serie
            preds_series = predictions_df[predictions_df["series_id"] == series_id]["pred"].reset_index(drop=True)
            
            if preds_series.empty:
                print(f"Warning: No predicted data for series_id {series_id}")
                plt.close(fig)
                continue
            
            # Índice temporal de validación (últimas n_val_weeks)
            val_idx = serie_real.index[-n_val_weeks:]
            
            # Verificar y ajustar longitudes si es necesario
            if len(preds_series) == len(val_idx):
                preds_series.index = val_idx
                preds_series.plot(ax=ax, color="orange", linewidth=2, 
                                label="Predictions")
                
                ax.axvspan(val_idx.min(), val_idx.max(), color="gray", alpha=0.1, label="Validation window")                
            else:
                print(f"Warning: Longitud inconsistente para {series_id}."
                      f"Predictions: {len(preds_series)}, Validation: {len(val_idx)}")
            
            # Configurar gráfico
            ax.set_title(f"Serie: {series_id}")
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error graficando serie {series_id}: {e}")
            plt.close(fig)



def plot_results_dual_scale(predictions_df, n_val_weeks, real_serie, specific_series=None, n_plots=4, col='ADD'):
    """
    Grafica los resultados con dos escalas: automática y fija (0-1).
    """
    figsize = (10, 3)
    series_to_plot = get_series_to_plot(real_serie, n_plots, specific_series)
    
    for series_id in series_to_plot:
        series_id = str(series_id)

        print(f"\nSerie: {series_id}")
        df_serie = (predictions_df.loc[predictions_df["series_id"].astype(str) == series_id,
                         ['y', "pred", "abs_error"]]
                    .sort_index())
        print(df_serie)
        print(f"MAE {df_serie['abs_error'].mean():.6f}")

        # Obtener datos
        preds_data = predictions_df[predictions_df["series_id"].astype(str) == series_id].copy()
        serie_real = real_serie[real_serie["series_id"] == series_id].copy()
        
        if preds_data.empty or serie_real.empty:
            print(f"Warning: No data available for series_id {series_id}")
            continue
            
        serie_real = serie_real.set_index("timestamp")[col].sort_index()
        
        try:
            # Preparar predicciones con timestamps correctos
            if 'timestamp' in preds_data.columns:
                preds_series = preds_data.set_index('timestamp')['pred'].sort_index()
            else:
                preds_series = preds_data['pred']
            
            # GRÁFICO 1: Escala automática
            fig, ax = plt.subplots(figsize=figsize)
            serie_real.plot(ax=ax, color="black", label="Real (train)")
            preds_series.plot(ax=ax, color="orange", linewidth=2, label="Predictions")
            
            if len(preds_series) > 0:
                val_start = preds_series.index.min()
                val_end = preds_series.index.max()
                ax.axvspan(val_start, val_end, color="gray", alpha=0.1, label="Validation window")
            
            ax.set_title(f"Serie: {series_id} (Escala Automática)")
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # GRÁFICO 2: Escala fija 0-1
            fig, ax = plt.subplots(figsize=figsize)
            serie_real.plot(ax=ax, color="black", label="Real (train)")
            preds_series.plot(ax=ax, color="orange", linewidth=2, label="Predictions")
            
            if len(preds_series) > 0:
                ax.axvspan(val_start, val_end, color="gray", alpha=0.1, label="Validation window")
            
            ax.set_ylim(-0.1, 1.1)  # Escala fija 0-1
            ax.set_title(f"Serie: {series_id} (Escala 0-1)")
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error graficando serie {series_id}: {e}")



# Tab 2: Residuals por horizonte

def create_residuals_plot(predictions_df, error_col='abs_error'):
    """
    Crea un boxplot de residuales por horizonte usando Plotly
    """
    df = predictions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Crear mapeo de horizonte
    snapshots = df["timestamp"].sort_values().unique()
    h_map = {ts: f"t+{i+1}" for i, ts in enumerate(snapshots)}
    df["horizon"] = df["timestamp"].map(h_map)
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    order = [f"t+{i+1}" for i in range(len(snapshots))]
    
    for horizon in order:
        data = df.loc[df["horizon"] == horizon, error_col].values
        fig.add_trace(go.Box(
            y=data,
            name=horizon,
            boxmean='sd'  # Muestra la media
        ))
    
    fig.update_layout(
        title="Residual distribution by horizon (grouped by snapshot)",
        xaxis_title="Horizon",
        yaxis_title=error_col,
        showlegend=False,
        height=500
    )
    
    # Calcular resumen estadístico
    resumen = (
        df.groupby("horizon")[error_col]
        .agg(n="count", mean="mean", median="median", std="std")
        .reindex(order)
    )
    
    return fig, resumen


# Tab 3: Line chart comparando real vs predicciones
def create_comparison_plot(predictions_df, real_serie, series_id, col='split'):
    """
    Crea un gráfico de línea comparando datos reales vs predicciones
    """
    series_id = str(series_id)
    
    # Filtrar datos de la serie específica
    preds_data = predictions_df[predictions_df["series_id"].astype(str) == series_id].copy()
    serie_real = real_serie[real_serie["series_id"] == series_id].copy()
    
    if preds_data.empty or serie_real.empty:
        return None, None
    
    # Preparar datos reales
    serie_real = serie_real.set_index("timestamp")[col].sort_index()
    
    # Preparar predicciones
    if 'timestamp' in preds_data.columns:
        preds_series = preds_data.set_index('timestamp')['pred'].sort_index()
    else:
        preds_series = preds_data['pred']
    
    # Calcular MAE
    mae = preds_data['abs_error'].mean()
    
    # Crear figura
    fig = go.Figure()
    
    # Datos reales (train)
    fig.add_trace(go.Scatter(
        x=serie_real.index,
        y=serie_real.values,
        mode='lines',
        name='Real (train)',
        line=dict(color='black')
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        x=preds_series.index,
        y=preds_series.values,
        mode='lines',
        name='Predictions',
        line=dict(color='orange', width=2)
    ))
    
    # Marcar ventana de validación
    if len(preds_series) > 0:
        val_start = preds_series.index.min()
        val_end = preds_series.index.max()
        
        fig.add_vrect(
            x0=val_start, x1=val_end,
            fillcolor="gray", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Validation window",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=f"Serie: {series_id} | MAE: {mae:.6f}",
        xaxis_title="timestamp",
        yaxis_title=col,
        hovermode='x unified',
        height=500
    )
    
    return fig, mae
