import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

def extract_date_from_filename(filename):
    """
    Extrae la fecha del nombre del archivo PSR_file_YYYY-MM-DD.xlsx
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d')
    return None

def read_series_from_snapshot(file_path):
    """
    Lee un archivo Excel y extrae los series_id únicos
    """
    try:
        df = pd.read_excel(file_path)
        
        # Buscar la columna de series_id (ajusta según tu estructura)
        series_column = 'Product ID'
                
        # Extraer series_id únicos (sin valores nulos)
        series_ids = set(df[series_column].dropna().unique())
        return series_ids
    
    except Exception as e:
        print(f"❌ Error leyendo {file_path.name}: {str(e)}")
        return set()

def build_series_status_table(data_folder):
    """
    Construye la tabla de trazabilidad de series
    
    Parameters:
    -----------
    data_folder : str o Path
        Carpeta donde están los archivos PSR_file_YYYY-MM-DD.xlsx
    output_file : str
        Nombre del archivo CSV de salida
    """
    
    data_path = Path(data_folder)
    
    # 1. Encontrar todos los archivos de snapshot
    snapshot_files = sorted(data_path.glob('PSR_file_*.xlsx'))
    
    if not snapshot_files:
        print(f"❌ No se encontraron archivos PSR_file_*.xlsx en {data_folder}")
        return None
    
    print(f"📁 Encontrados {len(snapshot_files)} archivos snapshot")
    
    # 2. Extraer fechas y series de cada snapshot
    snapshots = []
    for file in snapshot_files:
        snapshot_date = extract_date_from_filename(file.name)
        if snapshot_date:
            series_ids = read_series_from_snapshot(file)
            snapshots.append({
                'date': snapshot_date,
                'series': series_ids,
                'filename': file.name
            })
            print(f"  ✓ {file.name}: {len(series_ids)} series")
    
    # Ordenar por fecha
    snapshots.sort(key=lambda x: x['date'])
    
    if not snapshots:
        print("❌ No se pudieron procesar snapshots")
        return None
    
    # 3. Construir la tabla de status
    all_series = set()
    for snap in snapshots:
        all_series.update(snap['series'])
    
    print(f"\n📊 Total de series únicas encontradas: {len(all_series)}")
    
    # 4. Crear DataFrame con todas las combinaciones serie x fecha
    records = []
    
    for series_id in sorted(all_series):
        first_appearance = None
        last_appearance = None
        weeks_active = 0
        
        for snap in snapshots:
            snapshot_date = snap['date']
            is_present = series_id in snap['series']
            
            # Determinar el status
            if is_present:
                if first_appearance is None:
                    # Primera vez que aparece
                    status = 'new'
                    first_appearance = snapshot_date
                else:
                    status = 'active'
                
                last_appearance = snapshot_date
                weeks_active += 1
            else:
                if first_appearance is None:
                    # La serie aún no ha aparecido
                    continue
                else:
                    # La serie existía pero ya no está presente
                    status = 'inactive'
            
            records.append({
                'snapshot_date': snapshot_date,
                'series_id': series_id,
                'status': status,
                'first_seen': first_appearance,
                'last_seen': last_appearance,
                'weeks_active': weeks_active if is_present else weeks_active
            })
    
    # 5. Crear DataFrame final
    df_status = pd.DataFrame(records)
    
    # Ordenar por fecha y series_id
    df_status = df_status.sort_values(['snapshot_date', 'series_id']).reset_index(drop=True)
    
    # 6. Guardar resultado
    
    print(f"   Registros totales: {len(df_status):,}")
    print(f"   Período: {df_status['snapshot_date'].min()} a {df_status['snapshot_date'].max()}")
    
    # 7. Resumen estadístico
    print("\n📈 Resumen por status:")
    print(df_status.groupby('status').size())
    
    print("\n📊 Series por semanas activas:")
    weekly_summary = df_status[df_status['status'] != 'inactive'].groupby('series_id')['weeks_active'].max()
    print(f"   - Con < 4 semanas: {(weekly_summary < 4).sum()}")
    print(f"   - Con 4-12 semanas: {((weekly_summary >= 4) & (weekly_summary < 12)).sum()}")
    print(f"   - Con ≥ 12 semanas: {(weekly_summary >= 12).sum()}")
    
    return df_status

def get_active_series_for_date(df_status, target_date, min_weeks=4):
    """
    Obtiene las series activas para una fecha específica con suficiente historial
    
    Parameters:
    -----------
    df_status : DataFrame
        Tabla de series_status generada
    target_date : str o datetime
        Fecha objetivo (formato 'YYYY-MM-DD')
    min_weeks : int
        Mínimo de semanas activas requeridas
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    snapshot_data = df_status[df_status['snapshot_date'] == target_date]
    
    active_series = snapshot_data[
        (snapshot_data['status'].isin(['active', 'new'])) &
        (snapshot_data['weeks_active'] >= min_weeks)
    ]
    
    return active_series['series_id'].tolist()

def get_new_series_for_date(df_status, target_date):
    """
    Obtiene las series nuevas (cold-start) para una fecha específica
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    new_series = df_status[
        (df_status['snapshot_date'] == target_date) &
        (df_status['status'] == 'new')
    ]
    
    return new_series['series_id'].tolist()

# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================
import os

if __name__ == "__main__":
    
    # 1. Generar la tabla de series_status
    data_folder = "data/Raw Data"  # 📁 Ajusta esta ruta a tu carpeta de datos
    output_dir = "data"
    output_file='series_status.csv'
    output_path = os.path.join(output_dir, output_file)

    df_status = build_series_status_table(
        data_folder=data_folder)
    
    #df_status.to_csv(output_path, index=False)
    print(f"\n✅ Tabla generada exitosamente: {output_path}")

    
    # 2. Ejemplos de consultas útiles
    if df_status is not None:
        print("\n" + "="*60)
        print("🔍 EJEMPLOS DE CONSULTAS")
        print("="*60)
        
        # Ejemplo 1: Series activas para forecasting (≥ 4 semanas)
        latest_date = df_status['snapshot_date'].max()
        active = get_active_series_for_date(df_status, latest_date, min_weeks=4)
        print(f"\n1. Series aptas para forecasting ({latest_date}):")
        print(f"   Total: {len(active)} series con ≥ 4 semanas activas")
        
        # Ejemplo 2: Series nuevas (cold-start)
        new = get_new_series_for_date(df_status, latest_date)
        print(f"\n2. Series nuevas detectadas ({latest_date}):")
        print(f"   Total: {len(new)} series (requieren cold-start)")
        
        # Ejemplo 3: Series que desaparecieron
        inactive = df_status[
            (df_status['snapshot_date'] == latest_date) &
            (df_status['status'] == 'inactive')
        ]
        print(f"\n3. Series inactivas ({latest_date}):")
        print(f"   Total: {len(inactive)} series descontinuadas")
        
        # Ejemplo 4: Ver evolución de una serie específica
        if len(df_status) > 0:
            sample_series = df_status['series_id'].iloc[0]
            series_history = df_status[df_status['series_id'] == sample_series]
            print(f"\n4. Historia de ejemplo - Serie: {sample_series}")
            print(series_history[['snapshot_date', 'status', 'weeks_active']].head(10))
