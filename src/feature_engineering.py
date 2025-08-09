import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import numpy as np

def get_features(df):
    """
    Crea nuevas variables a partir de los datos existentes.
    Args:
        df (pd.DataFrame): DataFrame que contiene el conjunto de datos.
    Returns:
        pd.DataFrame: DataFrame con las nuevas características agregadas.
    """
    df_new = df.copy()

    # Features comerciales
    df_new['productos_por_compra'] =\
          (df_new['cantidad_productos_total_negocios_vendidos_mes'] / df_new['cantidad_compras_total_negocios_mes']).replace([np.inf, -np.inf], 0) #variedad de productos comprados por compra
    
    df_new['venta_promedio_x_compra'] =\
          (df_new['venta_total_negocios_mes'] / df_new['cantidad_compras_total_negocios_mes']).replace([np.inf, -np.inf], 0)  # $/n_compras
    
    df_new['venta_por_heladera'] = (df_new['venta_total_negocios_mes'] / df_new['cantidad_heladeras']).replace([np.inf, -np.inf], 0)   # $/n_heladeras

    df_new['freq_compra'] =( 1 / df_new['dias_entre_compras_total_negocios_mes'] ).replace([np.inf, -np.inf], 0) # frecuencia de compra (1/días entre compras)

    for i in range(1,5):
        df_new[f'ratio_neg{i}'] = (df_new[f'venta_negocio{i}_mes'] / df_new['venta_total_negocios_mes']).replace([np.inf, -np.inf], 0)  # ratio de venta por negocio
    

    # Features temporales
    df_new['fecha'] = pd.to_datetime(df['aniomes'], format='%Y%m')
    df_new['month'] = df_new['fecha'].dt.month
    df_new['mes_sin'] = np.sin(2 * np.pi * df_new['fecha'].dt.month / 12)
    df_new['mes_cos'] = np.cos(2 * np.pi * df_new['fecha'].dt.month / 12)
    df_new['is_first_month'] = (df_new['aniomes'] == 202404).astype('int')    # Primer mes registrado (valores faltantes debido a historial previo)

    # Features que dependen del cliente
    df_new['promedio_compras_cliente'] = df.groupby('cliente_id')['cantidad_compras_total_negocios_mes'].transform('mean') # Promedio de compras por mes del cliente

    df_new['promedio_productos_cliente'] = (
        df.groupby('cliente_id')['cantidad_productos_total_negocios_vendidos_mes'].transform('mean')) # Promedio de productos por compra (cliente)
    
    df_new['promedio_venta_total'] = (
        df.groupby('cliente_id')['venta_total_negocios_mes'].transform('mean')) # Promedio de venta total por mes (cliente)
    

    return df_new


def special_clients(df, feature, quantile=.99):
    """
    """
    for feature in feature:
        threshold = df[feature].quantile(quantile)
        df[f'is_especial_{feature}'] = (df[feature]>threshold).astype(int)
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




