from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import bayesian_search_forecaster_multiseries, OneStepAheadFold, TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.preprocessing import RollingFeatures

RANDOM_STATE = 42


### FORECASTER WITH HYPERPARAMETER FINE TUNING 

def create_forecaster(lags=[1,2,3,4]):
    """Create a LightGBM-based multi-series recursive forecaster."""
    lgbm = LGBMRegressor(
        random_state=RANDOM_STATE,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=lgbm,
        lags=lags,
        encoding="ordinal"
    )
    
    return forecaster


def tune_forecaster(series_dict_train, exog_dict_train=None, 
                   n_val_weeks=10, n_trials=20, metric="mean_absolute_error"):
    """Tune forecaster hyperparameters using Bayesian optimization."""
    
    # Calculate training window size
    n_total_weeks_train = len(next(iter(series_dict_train.values())))
    initial_train_size = n_total_weeks_train - n_val_weeks
    
    if initial_train_size <= 8:
        raise ValueError(f"Training window too small ({initial_train_size}). Reduce n_val_weeks.")
    
    # Define search space
    def search_space(trial):
        return {
            'lags': trial.suggest_categorical('lags', [
                [1,2,3,4],
                [1,2,3,4,8],
                [1,2,3,4,8,12],
                [1,2,3,4,8,12,24]
            ]),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 8, step=1),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63, step=4),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0, step=0.1),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0, step=0.1),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0, step=0.1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0, step=0.1),
        }
    
    # Create base forecaster
    forecaster = create_forecaster()
    
    # Setup cross-validation
    cv_osa = OneStepAheadFold(initial_train_size=initial_train_size)
    
    # Perform Bayesian search
    results_search, best_trial = bayesian_search_forecaster_multiseries(
        forecaster=forecaster,
        series=series_dict_train,
        exog=exog_dict_train,
        cv=cv_osa,
        search_space=search_space,
        n_trials=n_trials,
        metric=metric,
        return_best=True,
        suppress_warnings=True,
        random_state=RANDOM_STATE,
        n_jobs="auto",
        verbose=True
    )
    
    # Extract best parameters
    best_params = results_search.at[0, 'params']
    best_lags = results_search.at[0, 'lags']
    
    print(f"Best lags: {best_lags}")
    print(f"Best params: {best_params}")
    
    return best_params, best_lags



### VALIDATION WITH BACKTESTING


def create_final_forecaster(best_params, best_lags):
    """Create final forecaster with optimized parameters and rolling features."""
    regressor = LGBMRegressor(
        random_state=RANDOM_STATE,
        **best_params
    )
    
    rolling = RollingFeatures(
        stats=['mean', 'mean'],
        window_sizes=[4, 7]
    )
    
    final_forecaster = ForecasterRecursiveMultiSeries(
        regressor=regressor, 
        lags=best_lags,
        window_features=rolling,
        encoding="ordinal", 
        dropna_from_series=False
    )
    
    return final_forecaster


def backtest_forecaster(forecaster, series_dict_train, exog_dict_train=None, 
                       steps=4, n_folds=3, metrics=["mean_absolute_error"]):
    """Perform backtesting on the forecaster using TimeSeriesFold cross-validation."""
    
    # Calculate validation parameters
    n_val_weeks_tsf = steps * n_folds
    n_total_weeks_train = len(next(iter(series_dict_train.values())))
    initial_train_size = n_total_weeks_train - n_val_weeks_tsf
    
    if initial_train_size <= 8:
        raise ValueError(f"Training window too small ({initial_train_size}). Reduce steps or n_folds.")
    
    print(f"Backtesting setup:")
    print(f"- Steps: {steps}")
    print(f"- Folds: {n_folds}")
    print(f"- Validation weeks: {n_val_weeks_tsf}")
    print(f"- Initial train size: {initial_train_size}")
    
    # TimeSeriesFold cross-validation
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=initial_train_size,
        refit=True,
        fixed_train_size=True,
        allow_incomplete_fold=True
    )
    
    # Perform backtesting
    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        series=series_dict_train,
        exog=exog_dict_train,
        cv=cv,
        levels=None,
        metric=metrics,
        add_aggregated_metric=True,
        n_jobs="auto",
        verbose=True,
        show_progress=True,
        suppress_warnings=True
    )
    
    metrics_levels = metrics_levels.rename(columns={"levels": "series_id"})
    backtest_predictions = backtest_predictions.rename(columns={"level": "series_id"})

    print("\nBacktesting Results:")
    print(metrics_levels)
    print("\nPredictions Sample:")
    print(backtest_predictions.head())
    
    return metrics_levels, backtest_predictions



### TESTING

def fit_predict(forecaster, series_dict_train, series_dict_test,
                     exog_dict_train=None, exog_dict_test=None, steps=4):
    
    # Entrenar
    forecaster.fit(series=series_dict_train, exog=exog_dict_train)
    
    # Predecir
    levels_test = list(series_dict_test.keys())
    predictions = forecaster.predict(
        steps=steps,
        levels=levels_test,
        exog=exog_dict_test
    )
    
    predictions = predictions.rename(columns={"level": "series_id"})

    return predictions