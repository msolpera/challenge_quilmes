from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd 
import numpy as np
from scipy.stats import skew



def pipeline_preprocesamiento_lr(numeric_cols, categorical_cols, 
                                     categorical_strategy='onehot', except_features=None, fit=False, X=None, y=None):
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

    # Features numéricas: imputar + transformar 
    num_pipeline = Pipeline([
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
        ('num', num_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    if fit and X is not None and y is not None:
        preprocessor.fit(X, y)

    return preprocessor


def preprocessor_cb(cat_features, num_features, fit=False, X=None, y=None):
    # Para numéricas: imputación por mediana
    num_prepro = Pipeline([
        ("imputation_none", SimpleImputer(missing_values=np.nan, strategy="median", add_indicator=True))
    ])
    
    # Para categóricas: sin indicadores extra
    cat_prepro = Pipeline([
        ("imputation", SimpleImputer(missing_values=np.nan,
                                              strategy='constant',
             fill_value='NA', add_indicator=True))
    ])
    feature_eng = ColumnTransformer([
        ("cat", cat_prepro, cat_features),
        ("num", num_prepro, num_features),
    ],
    remainder="passthrough",  
    verbose=False,
    verbose_feature_names_out=True)

    preprocessor = Pipeline([("feature_eng", feature_eng)])

    if fit and X is not None and y is not None:
        preprocessor.fit(X, y)
    return preprocessor