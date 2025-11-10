# Required columns on input dataframe: 
#['timestamp', 'series_id', 'Brand', 'Resource ID', 'ADD', 'sin_week', 'cos_week', 'week_number']


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def load_data(filepath):
    """Load and prepare series and exog dataframes from Excel file."""
    features = pd.read_excel(filepath)
    return features


def series_exog(df):
    cols = df.columns.tolist()
    
    exog = df[[cols[0], cols[1], 'sin_week', 'cos_week', 'week_number']].copy()  

    rest = [c for c in df.columns if c not in exog.columns and c not in (cols[0], cols[1])]
    series = df[[cols[0], cols[1], *rest]].copy()

    series['timestamp'] = pd.to_datetime(series['timestamp'])
    exog['timestamp'] = pd.to_datetime(exog['timestamp'])
    
    return series, exog



def train_test_split(series, exog, test_weeks=4):
    """Split time series data into train/test sets by date."""
    last_date = series['timestamp'].max()
    end_train = last_date - pd.Timedelta(weeks=test_weeks)
    start_test = end_train + pd.Timedelta(weeks=1)
    
    series_train = series.loc[series['timestamp'] <= end_train].copy()
    series_test = series.loc[series['timestamp'] >= start_test].copy()
    exog_train = exog.loc[exog['timestamp'] <= end_train].copy()
    exog_test = exog.loc[exog['timestamp'] >= start_test].copy()
    
    return series_train, series_test, exog_train, exog_test



def calculate_stats_train(series_train, column='ADD'):
    """Calculate training statistics (mean and std) by series_id."""
    stats_train = (
        series_train
        .groupby('series_id')[column]
        .agg(mu='mean', sigma=lambda x: x.std(ddof=0))
        .reset_index()
    )
    stats_train['sigma'] = stats_train['sigma'].replace(0.0, 1.0)
    
    return stats_train


def normalize_series(series_train, series_test, stats_train=None, column='ADD'):
    """Normalize series using pre-calculated training statistics."""
    if stats_train is None:
        stats_train = calculate_stats_train(series_train, column)
        
    series_train = series_train.merge(stats_train[['series_id', 'mu', 'sigma']],
                                      on='series_id', how='left')
    series_test = series_test.merge(stats_train[['series_id', 'mu', 'sigma']],
                                    on='series_id', how='left')
    
    z_col = f'z_{column}'
    series_train[z_col] = (series_train[column] - series_train['mu']) / series_train['sigma']
    series_test[z_col] = (series_test[column] - series_test['mu']) / series_test['sigma']
    
    return series_train, series_test




def create_time_series_dicts(series_train, series_test, exog_train, exog_test, 
                             target_col='z_ADD', freq='W-WED'):
    """Convert train/test dataframes to time series dictionaries."""
    # Sort data
    series_train = series_train.sort_values(['series_id', 'timestamp'])
    series_test = series_test.sort_values(['series_id', 'timestamp'])
    exog_train = exog_train.sort_values(['series_id', 'timestamp'])
    exog_test = exog_test.sort_values(['series_id', 'timestamp'])
    
    # Create series dictionaries
    series_dict_train = {
        sku: df.set_index('timestamp')[target_col].sort_index().asfreq(freq)
        for sku, df in series_train.groupby('series_id', sort=False)
    }
    
    series_dict_test = {
        sku: df.set_index('timestamp')[target_col].sort_index().asfreq(freq)
        for sku, df in series_test.groupby('series_id', sort=False)
    }
    
    # Create exog dictionaries
    exog_dict_train = {
        sku: df.set_index('timestamp').drop(columns=['series_id']).sort_index().asfreq(freq)
        for sku, df in exog_train.groupby('series_id', sort=False)
    }
    
    exog_dict_test = {
        sku: df.set_index('timestamp').drop(columns=['series_id']).sort_index().asfreq(freq)
        for sku, df in exog_test.groupby('series_id', sort=False)
    }
    
    return series_dict_train, series_dict_test, exog_dict_train, exog_dict_test




def get_series_preds(predictions, series_test):
    keys = ['timestamp', 'series_id']

    pred_map = (predictions.drop_duplicates(subset=keys, keep='last')
                            .set_index(keys)['pred'])
    pred_map = pred_map.mask(pred_map < 0, 0)

    series_preds = series_test[['timestamp', 'series_id', 'ADD', 'Brand', 'Resource ID']].copy().set_index(keys)
    series_preds['ADD'] = pred_map.reindex(series_preds.index).values
    series_preds = series_preds.reset_index()
    series_preds = series_preds.rename(columns = {'ADD': 'pred'})
    return series_preds


def get_split_col(df, demand_col="ADD", time_col="timestamp",
                           brand_col="Brand", resource_col="Resource ID"):
    """
    Versión que retorna solo df_pairs para casos donde no necesitas el DataFrame completo.
    Input: timestamp	series_id(sku)	ADD	 Brand	Resource ID
    Output: timestamp	series_id	ADD	Brand	Resource ID	ADD_rec	ADD_brand	split (fraccion)
    """
    
    out = df.copy()
    
    out["ADD_rec"] = out.groupby([time_col, brand_col, resource_col])[demand_col].transform("sum")
    
    brand_sums = (out.drop_duplicates([time_col, brand_col, resource_col])
                  .groupby([time_col, brand_col])["ADD_rec"].sum())
    
    out["ADD_brand"] = out.set_index([time_col, brand_col]).index.map(brand_sums).values
    
    out["split"] = np.where(
        out["ADD_brand"] > 0,
        out["ADD_rec"] / out["ADD_brand"],
        1 / out.groupby([time_col, brand_col])[resource_col].transform("nunique")
    )
        # Crear df_pairs directamente
    df_pairs = (
        out[[time_col, brand_col, resource_col, 'split']]
        .drop_duplicates([time_col, brand_col, resource_col])
        .sort_values([time_col, brand_col, resource_col])
        .reset_index(drop=True)
    )
    
    # Añadir identificadores
    df_pairs['series_id'] = (df_pairs[brand_col].astype('string') + '||' + 
                          df_pairs[resource_col].astype('string'))
    return df_pairs




def get_df_input(df_pairs, demand_col="ADD", time_col="timestamp", brand_col="Brand", resource_col="Resource ID"):
    '''
    Input: timestamp	series_id	ADD	Brand	Resource ID	ADD_rec	ADD_brand	split
    Output: timestamp	series_id	split
    '''
 
    df_input = df_pairs.loc[:, ['timestamp','series_id','split']].copy()
    eps=1e-6
    p = df_input['split'].astype(float).clip(eps, 1 - eps)
    df_input['s_split'] = np.log(p / (1 - p))

    return df_input



### Pendiente extender para ADD, con stats
def get_full_data(series, exog, target_col):
    #if target_col == 'ADD':
    series_full_dict = {
        sku: df.set_index('timestamp')[target_col].sort_index().asfreq('W-WED')
        for sku, df in series.groupby('series_id', sort=False)
    }

    exog_full_dict = {
        sku: df.set_index('timestamp').drop(columns=['series_id']).sort_index().asfreq('W-WED')
        for sku, df in exog.groupby('series_id', sort=False)
    }

    return series_full_dict, exog_full_dict


def get_future_exog(horizon, series_full):
    series_ids = series_full['series_id'].unique()

    last_ts = series_full['timestamp'].max()
    future_ts = [last_ts + pd.Timedelta(days=7*i) for i in range(1, horizon + 1)]
    future_ts = pd.to_datetime(future_ts)

    base = pd.DataFrame({'timestamp': future_ts})
    base['weekofyear'] = base['timestamp'].dt.isocalendar().week
    base['sin_week'] = np.sin(2 * np.pi * base['weekofyear'] / 52)
    base['cos_week'] = np.cos(2 * np.pi * base['weekofyear'] / 52)

    week_lookup = {date: idx+1 for idx, date in enumerate(sorted(base['timestamp'].unique()))}
    base['week_number'] = base['timestamp'].map(week_lookup)

    future_exog_dict = {
        sid: base.assign(series_id=sid)
                .set_index('timestamp')[['series_id', 'sin_week', 'cos_week', 'week_number']]
        for sid in series_ids}
    
    return future_exog_dict

def get_final_df(f_predictions, series_test):
    series_preds = get_series_preds(f_predictions, series_test)
    split_preds = get_split_col(series_preds, demand_col="pred")
    split_test = get_split_col(series_test)
    final_df = split_preds[['timestamp', 'series_id', 'split']].copy()
    final_df = final_df.rename(columns= {'split': 'pred'})
    
    final_df = final_df.merge(
            split_test[["timestamp", "series_id", "split"]],
            on=["timestamp", "series_id"],
            how="left"
        ).rename(columns={'split': 'y'})
    
    final_df["abs_error"] = (final_df["pred"] - final_df["y"]).abs()
        
    mae_by_series = final_df.groupby('series_id')['abs_error'].mean()
    final_df['mae'] = final_df['series_id'].map(mae_by_series)
    return final_df