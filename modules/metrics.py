import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt


def format_results(predictions, series_dict, scenario_data, scenario_type):
    """
    Format the predictions dataframe by creating all the necessary columns.
    """
    scale = 'z' if scenario_type == 'v1' else 's'

    pred_scale = f"pred_{scale}"
    y_scale = f"y_{scale}"
    abs_error_scale = f"abs_error_{scale}"
    mae_scale = f"mae_{scale}"

    predictions = predictions.copy()
    
    predictions.index = pd.to_datetime(predictions.index)
    predictions.index.name = "timestamp"
    predictions = predictions.reset_index()

    # scaled predictions (z score // logit-sigmoid)
    predictions = predictions.rename(columns={"pred": pred_scale})

    def get_real_values(df, series_dict):
        real_values = []
        for _, row in df.iterrows():
            series_id = row["series_id"]
            timestamp = row["timestamp"]
            try:
                real_values.append(series_dict[series_id].loc[timestamp])
            except (KeyError, IndexError):
                real_values.append(None)
        return real_values
    
    predictions[y_scale] = get_real_values(predictions, series_dict)

    predictions[abs_error_scale] = (predictions[pred_scale] - predictions[y_scale]).abs()

    scale_mae_by_series = predictions.groupby('series_id')[abs_error_scale].mean()
    predictions[mae_scale] = predictions['series_id'].map(scale_mae_by_series)

    # original scale
    if scenario_type == 'v1':
        predictions = _apply_v1_transform(predictions, scenario_data, pred_scale, y_scale)
    elif scenario_type == 'v2':
        predictions = _apply_v2_transform(predictions, scenario_data, pred_scale)


    predictions["abs_error"] = (predictions["pred"] - predictions["y"]).abs()
    
    mae_by_series = predictions.groupby('series_id')['abs_error'].mean()
    predictions['mae'] = predictions['series_id'].map(mae_by_series)

    scale_mae_global = predictions[mae_scale].mean()
    mae_global = predictions["mae"].mean()

    
    columns_to_return = ['timestamp', 'series_id', 
                         pred_scale, y_scale, abs_error_scale, mae_scale, 
                        'pred', 'y', 'abs_error', 'mae']
    
    return predictions[columns_to_return], scale_mae_global, mae_global


### Auxiliar functions of format_results()
def _apply_v1_transform(predictions, stats_train, pred_scale, y_scale):
    """Aplicar transformación v1 (z-score)"""
    predictions = predictions.merge(
        stats_train[["series_id", "sigma", "mu"]],
        on="series_id",
        how="left"
    )
    
    predictions["pred"] = (predictions[pred_scale] * predictions["sigma"] + predictions["mu"]).clip(lower=0)
    predictions["y"] = predictions[y_scale] * predictions["sigma"] + predictions["mu"]
    
    return predictions


def _apply_v2_transform(predictions, df_full, pred_scale):
    """Aplicar transformación v2 (logit-sigmoid)"""
    predictions = predictions.merge(
        df_full[["series_id", "timestamp", "Brand", "split"]].rename(columns={"split": "y"}),
        on=["series_id", "timestamp"],
        how="left"
    )
    
    # Transformación sigmoid y normalización
    predictions['p_raw'] = 1.0 / (1.0 + np.exp(-predictions[pred_scale].astype(float)))
    den = predictions.groupby(['timestamp', 'Brand'])['p_raw'].transform('sum')
    predictions['pred'] = np.where(den > 0, predictions['p_raw'] / den, 0.0)
    
    return predictions





def mean_ci_t_by_group(df, group_col, value_col, alpha=0.05):
    """
    Calculate CI of errors (original and z scale)
    """
    g = df.groupby(group_col)[value_col]
    out = g.agg(n='count', mean='mean', std=lambda x: x.std(ddof=1)).reset_index()
    out['se'] = out['std'] / np.sqrt(out['n'])
    out['df'] = out['n'] - 1
    out['tcrit'] = out['df'].apply(lambda dfi: stats.t.ppf(1 - alpha/2, dfi) if dfi>0 else np.nan)
    out['ci_lo'] = out['mean'] - out['tcrit'] * out['se']
    out['ci_hi'] = out['mean'] + out['tcrit'] * out['se']
    out = out.rename(columns={
        'mean': f'{value_col}_mae',
        'std': f'{value_col}_std',
        'se':  f'{value_col}_se',
        'ci_lo': f'{value_col}_ci_lo',
        'ci_hi': f'{value_col}_ci_hi'
    })
    return out[[group_col, 'n', 'df', f'{value_col}_mae', f'{value_col}_std',
                f'{value_col}_se', 'tcrit', f'{value_col}_ci_lo', f'{value_col}_ci_hi']]


def summary_mae_ci(per_sku_df, mae_col, alpha=0.05):
    '''
    Macro-MAE (promedio de los MAE por SKU) con IC t entre SKUs
    '''
    vals = per_sku_df[mae_col].dropna().values
    S = len(vals)
    mean_macro = vals.mean()
    std_between = vals.std(ddof=1) if S>1 else np.nan
    se_between = std_between / np.sqrt(S) if S>1 else np.nan
    tcrit = stats.t.ppf(1 - alpha/2, S-1) if S>1 else np.nan
    ci_lo = mean_macro - tcrit*se_between if S>1 else np.nan
    ci_hi = mean_macro + tcrit*se_between if S>1 else np.nan
    return pd.Series({'S': S, 'mean_macro': mean_macro, 'std_between': std_between,
                      'se_between': se_between, 'tcrit': tcrit, 'ci_lo': ci_lo, 'ci_hi': ci_hi})


def get_full_metrics(predictions, scenario_type):

    scale = 'z' if scenario_type == 'v1' else 's'

    # --- IC por SKU, en original y en z ---
    ci_orig = mean_ci_t_by_group(predictions, 'series_id', 'abs_error')
    ci_scale    = mean_ci_t_by_group(predictions, 'series_id', f'abs_error_{scale}')

    # Merge de ambas escalas
    per_sku = ci_orig.merge(ci_scale, on=['series_id','n','df','tcrit'], how='inner')

    macro_orig = summary_mae_ci(per_sku, 'mae')
    macro_z    = summary_mae_ci(per_sku, f'mae_{scale}')

    print("\nMacro-MAE (original):")
    print(macro_orig)
    print("\nMacro-MAE (scale):")
    print(macro_z)


def residuals(df, error_col = 'abs_error'):
    predictions = df.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    snapshots = predictions["timestamp"].sort_values().unique()
    h_map = {ts: f"t+{i+1}" for i, ts in enumerate(snapshots)}
    predictions["horizon"] = predictions["timestamp"].map(h_map)

    order = [f"t+{i+1}" for i in range(len(snapshots))]

    data = [predictions.loc[predictions["horizon"] == h, error_col].values for h in order]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(order) + 1), order)
    plt.title("Residual distribution by horizon (grouped by snapshot)")
    plt.xlabel("Horizon")
    plt.ylabel(f"{error_col}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    resumen = (
        predictions.groupby("horizon")[error_col]
        .agg(n="count", mean="mean", median="median", std="std")
        .reindex(order)
    )
    print(resumen) 



def logit_post(df):
    eps = 1e-6
    y = df["y"].astype(float).clip(eps, 1-eps)
    pred = df["pred"].astype(float).clip(eps, 1-eps)

    logit = lambda p: np.log(p/(1-p))
    mae_logit_post = np.mean(np.abs(logit(pred) - logit(y)))

    return mae_logit_post

    
