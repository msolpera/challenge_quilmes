import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import numpy as np

def features_eng(df):
    """
    Crea nuevas variables a partir de los datos existentes.
    Args:
        df (pd.DataFrame): DataFrame que contiene el conjunto de datos.
    Returns:
        pd.DataFrame: DataFrame con las nuevas características agregadas.
    """
    df_new = df.copy()

    # Features comerciales
    den = df_new['cantidad_compras_total_negocios_mes']
    mask_cero = den == 0
    result = np.full(len(df_new), np.nan)
    result[~mask_cero] = df_new.loc[~mask_cero, 'cantidad_productos_total_negocios_vendidos_mes'] / den[~mask_cero]
    result[mask_cero] = 0
    df_new['productos_por_compra'] = result
    
    # Para 'venta_promedio_x_compra'
    den = df_new['cantidad_compras_total_negocios_mes']
    mask_cero = den == 0
    result = np.full(len(df_new), np.nan)
    result[~mask_cero] = df_new.loc[~mask_cero, 'venta_total_negocios_mes'] / den[~mask_cero]
    result[mask_cero] = 0
    df_new['venta_promedio_x_compra'] = result
    
    # Para 'venta_por_heladera'
    den = df_new['cantidad_heladeras']
    mask_cero = den == 0
    result = np.full(len(df_new), np.nan)
    result[~mask_cero] = df_new.loc[~mask_cero, 'venta_total_negocios_mes'] / den[~mask_cero]
    result[mask_cero] = 0
    df_new['venta_por_heladera'] = result
    
    # Para 'freq_compra' (1 / días_entre_compras_total_negocios_mes)
    den = df_new['dias_entre_compras_total_negocios_mes']
    mask_cero = den == 0
    result = np.full(len(df_new), np.nan)
    result[~mask_cero] = 1 / den[~mask_cero]
    result[mask_cero] = 0
    df_new['freq_compra'] = result
    
    # Para ratios negocio i
    for i in range(1, 5):
        den = np.abs(df_new['venta_total_negocios_mes'])
        mask_cero = den == 0
        result = np.full(len(df_new), np.nan)
        result[~mask_cero] = df_new.loc[~mask_cero, f'venta_negocio{i}_mes'] / den[~mask_cero]
        result[mask_cero] = 0
        df_new[f'ratio_neg{i}'] = result

    # suma de ventas == 0
    suma_ventas = df_new[[f'venta_negocio{i}_mes' for i in range(1, 5)]].sum(axis=1) + df_new['venta_total_negocios_mes']
    df_new['flag_suma_ventas_0'] = (suma_ventas == 0).astype(int)

    return df_new


def agregar_flag_outlier(df, ventas_col='venta_total_negocios_mes', fecha_col='aniomes', quantil=0.99):
    """
    Agrega una columna de flag para identificar outliers en las ventas.

    """
    df = df.copy()

    umbrales = df.groupby(fecha_col)[ventas_col].quantile(quantil).to_dict()

    def flag_outlier(row):
        umbral_mes = umbrales.get(row[fecha_col], 0)
        return int(row[ventas_col] > umbral_mes)
    
    df['flag_outlier'] = df.apply(flag_outlier, axis=1)
    return df

def log_transform(df, feature):
    """
    """
    df = df.copy()
    for col in feature:
        df[f'{col}_log'] = np.log1p(df[col])
    return df


def get_cat_num_features(X_train):
    """
    Obtiene las variables categóricas y numéricas del conjunto de entrenamiento.
    Args:
        X_train (pd.DataFrame): Conjunto de entrenamiento.
    Returns:
        tuple: (cat_features, num_features) donde:
            - cat_features (list): Lista de nombres de variables categóricas.
            - num_features (list): Lista de nombres de variables numéricas.
    """
    cat_features = X_train.select_dtypes(include=['object']).columns.to_list()
    num_features = X_train.select_dtypes(include=['number']).columns.to_list()
    print(f'{len(cat_features)} variables categóricas')
    print(f'{len(num_features)} variables numéricas')
    print(f'{len(num_features + cat_features)} variables en total')

    return cat_features, num_features


