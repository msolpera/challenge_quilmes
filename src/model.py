from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import numpy as np
from feature_engineering import get_cat_num_features
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import pandas as pd
from utils import error_ponderado_negocio


def split_data(df, fecha_corte_train=202504,fecha_oot=202505):
    """
    Splits the dataset into training, validation and testing sets.

    Args:
        df (pd.DataFrame): The complete dataset including features and target.
        target_col (str): The name of the target variable column.
        valid_size (float): Number of months to include in the validation set.
        test_size (float): Number of months to include in the test set.

    Returns:
        tuple: df_train, df_valid, df_test
    """

    df_train = df[df['aniomes'] < fecha_corte_train]
    df_valid = df[df['aniomes'] == fecha_corte_train]
    df_test = df[df['aniomes'] == fecha_oot]

    return df_train, df_valid, df_test


def train_model(X_train, y_train, model_name, preprocessor=None):
    """
    Entrena un modelo de regresión lineal o CatBoost.
    """

    if model_name == 'linear':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

    elif model_name == 'catboost':
    
        cb = CatBoostRegressor(
            iterations=424,
            learning_rate=0.05,
            depth=10,
            loss_function='MAE',
            eval_metric='MAE',
            random_state=42,
            verbose=False
            )
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('catboost', cb)
        ])
        # Obtener nombres de columnas tras preprocesamiento
        feature_names = preprocessor.named_steps['feature_eng'].get_feature_names_out()
        
        # Detectar índices de columnas categóricas
        cat_feature_indices = [i for i, name in enumerate(feature_names) if name.startswith('cat__')]
    
    else:
        print(f"Modelo {model_name} no reconocido. Usa 'linear' o 'catboost'.")
        
    # Entrenar modelo
    if model_name == 'catboost':
        model.fit(X_train, y_train,  catboost__cat_features=cat_feature_indices)
    else:
        model.fit(X_train, y_train)

    # Guardar
    import joblib
    filename = f"modelo_{model_name}.pkl"
    joblib.dump(model, filename)
    print(f"Modelo guardado en {filename}")

    return model



def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using MAE and RMSE (with confidence intervals).
    Also compares it against a DummyRegressor baseline.

    Args:
        model (object): Trained regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values.
        baseline (bool): Whether to compare against a DummyRegressor.

    Returns:
        None
    """
    y_test = y_test
    y_pred = model.steps[-1][1].predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # MAE and RMSE using bootstrap
    mae_ci = bootstrap(y_test.values, y_pred, mean_absolute_error)
    rmse_ci = bootstrap(y_test.values, y_pred, lambda y1, y2: root_mean_squared_error(y1, y2))


    print("\n performance on test:")
    print("----------------------------------")
    print(" Model Evaluation Summary:")
    print("----------------------------------")
    print(f" MAE: {mae:,.2f} min ")
    print(f"95% Confidence Interval for MAE: {mae_ci[0]:,.2f} min – {mae_ci[1]:,.2f} min\n")

    print(f"RMSE: {rmse:,.2f} min ")
    print(f"95% Confidence Interval for RMSE: {rmse_ci[0]:,.2f} min – {rmse_ci[1]:,.2f} min\n")

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_test, y_test)
    dummy_preds = dummy.predict(X_test)
    dummy_mae = mean_absolute_error(y_test, dummy_preds)
    dummy_rmse = root_mean_squared_error(y_test, dummy_preds)
    dummy_r2 = r2_score(y_test, dummy_preds)


        

    improvement_mae = dummy_mae - mae
    improvement_rmse = dummy_rmse - rmse

    print("Comparison vs baseline (dum. regressor using mean):")
    print("--------------------------------------------------------")
    print(f" MAE (Dummy): {dummy_mae:,.2f} min -  improvement: {improvement_mae:,.2f} min")
    print(f" RMSE (Dummy): {dummy_rmse:,.2f} min  -  improvement: {improvement_rmse:,.2f} min")
    return mae, rmse, r2, dummy_mae, dummy_rmse, dummy_r2



def bootstrap(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """
    Computes a confidence interval for a metric using bootstrap resampling.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
        metric_fn (function): Metric function to apply (e.g., mean_absolute_error).
        n_bootstrap (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (e.g., 95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """

    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)

    return lower, upper


def weighted_mae(y_true, y_pred, sample_weight):
    return np.sum(sample_weight * np.abs(y_true - y_pred)) / np.sum(sample_weight)

