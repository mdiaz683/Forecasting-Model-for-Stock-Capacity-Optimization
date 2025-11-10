# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.features import load_data, series_exog, train_test_split, calculate_stats_train, normalize_series, create_time_series_dicts, get_split_col, get_df_input

from modules.plotting import plot_series, plot_train_test_split, plot_results, plot_results_dual_scale

from modules.lgbm_architecture import create_forecaster, tune_forecaster, create_final_forecaster, backtest_forecaster, fit_predict

from modules.metrics import format_results, residuals, get_full_metrics

STEPS = 4
filepath="./features_py.xlsx"

features = load_data(filepath)

series_ini, exog_ini = series_exog(features)

df_full = get_split_col(series_ini)


df_input = get_df_input(df_full)

wk_feats = (exog_ini[['timestamp', 'sin_week', 'cos_week', 'week_number']]
            .drop_duplicates(subset=['timestamp']))
df_input = df_input.merge(wk_feats, on='timestamp', how='left')

series, exog = series_exog(df_input)

series_train, series_test, exog_train, exog_test = train_test_split(series, exog, test_weeks=STEPS)

series_dict_train, series_dict_test, exog_dict_train, exog_dict_test = create_time_series_dicts(
    series_train, series_test, exog_train, exog_test, target_col='s_split'
)

forecaster = create_forecaster()

best_params, best_lags = tune_forecaster(series_dict_train, exog_dict_train)

final_forecaster = create_final_forecaster(best_params, best_lags)

#metrics_levels, backtest_predictions = backtest_forecaster(final_forecaster, 
#                                                           series_dict_train, exog_dict_train,
#                                                           steps=STEPS)

# f_backtest_predictions = format_results(backtest_predictions, series_dict_train, df_full, 'v2')

predictions = fit_predict(final_forecaster, 
                          series_dict_train, series_dict_test, 
                          exog_dict_train= exog_dict_train, exog_dict_test=exog_dict_test,
                          steps=STEPS)

f_predictions = format_results(predictions,
                               series_dict_test,
                               df_full,
                               'v2')

print(f_predictions)