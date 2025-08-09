from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd 
import numpy as np
from scipy.stats import skew



def pipeline_preprocesamiento_rdmforest(numeric_cols, categorical_cols, 
                                     categorical_strategy='onehot', except_features=None):
    """
    Pipeline de preprocesamiento para variables numéricas y categóricas.

    Args:
        numeric_cols (list): Lista de columnas numéricas.
        categorical_cols (list): Lista de columnas categóricas.
        categorical_strategy (str): 'onehot'.

    Returns:
        sklearn Pipeline: Pipeline de preprocesamiento.
    """
    numeric_cols = [col for col in numeric_cols if col not in (except_features or [])]

    # Features numéricas: imputar + transformar + escalar
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Features categóricas: imputar + codificar
    if categorical_strategy == 'onehot':
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

    elif categorical_strategy == 'ordinal':
        from sklearn.preprocessing import OrdinalEncoder
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    else:
        raise ValueError("categorical_strategy debe ser 'onehot' o 'ordinal'")

    # Combinador de columnas
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def pipeline_preprocesamiento_catboost(numeric_cols, categorical_cols, except_features=None):
    """
    Pipeline de preprocesamiento para CatBoost.
    Args:
        numeric_cols (list): Lista de columnas numéricas.
        categorical_cols (list): Lista de columnas categóricas.
        except_features (list, optional): Columnas a excluir del preprocesamiento. Defaults to None.
        fit (bool, optional): Si True, ajusta el preprocesador a los datos. Defaults to False.
        X (pd.DataFrame, optional): Datos de entrada para ajustar el preprocesador. Defaults to None.
        y (pd.Series, optional): Etiquetas para ajustar el preprocesador. Defaults to None.
        Returns:    
        sklearn Pipeline: Pipeline de preprocesamiento."""
    if except_features is None:
        except_features = []

    numeric_cols = [col for col in numeric_cols if col not in except_features]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA'))
    ])

    feature_eng = ColumnTransformer([
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ], remainder='drop', verbose=False)

    preprocessor = Pipeline([("feature_eng", feature_eng)])


    return preprocessor

    