import pandas as pd
import numpy as np


def get_features(df):
    """Feature engineering: create new features based on existing data.
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.    
    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()

    # Features comerciales
    df_new['ventas_por_visita'] =\
        (df_new['venta_total_negocios_mes'] / df_new['minutos_visitados_mes']).fillna(0)  # # $/minuto en cada visita
    
    df_new['productos_por_compra'] =\
          (df_new['cantidad_productos_total_negocios_vendidos_mes'] / df_new['cantidad_compras_total_negocios_mes']).fillna(0) #variedad de productos comprados por compra
    
    df_new['venta_promedio_x_compra'] =\
          (df_new['venta_total_negocios_mes'] / df_new['cantidad_compras_total_negocios_mes']).fillna(0)  # $/n_compras

    # Features temporales
    df_new['fecha'] = pd.to_datetime(df['aniomes'], format='%Y%m')
    df_new['mes_sin'] = np.sin(2 * np.pi * df_new['fecha'].dt.month / 12)
    df_new['mes_cos'] = np.cos(2 * np.pi * df_new['fecha'].dt.month / 12)
    df_new['is_first_month'] = df_new['aniomes'] == 202404  # Primer mes registrado (valores faltantes debido a historial previo)

    return df_new

