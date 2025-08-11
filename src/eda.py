import pandas as pd
import numpy as np
import plotly.express as px

def describe_num_cols(df, num_features):
    """
    Display statistical summary and correlation matrix
    
    Args:
        df (pd.DataFrame): Input DataFrame
        num_features (list): List of numerical column names
    """
    print(f"\nStatistical data of the numerical columns:")
    print("="*50)
    print(df[num_features].describe().T.round(3))
    
    print(f"\nCorrelation Matrix:")
    print("="*50)

    cols = num_features.copy()
    
    # Crear heatmap
    fig = px.imshow(
        df[cols].corr(),
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.show()

def resumen_unique_values(df, exclude=['cliente_id']):
    """
    Resumen de la cantidad y los valores únicos por variable categórica.

    Args:
    df (pd.DataFrame): DataFrame de entrada.
    exclude (list): Lista de columnas a excluir del análisis.

    Returns:
    pd.DataFrame: Tabla con columnas categóricas, cantidad de valores únicos y los valores.
    """
    # Seleccionar variables categóricas distintas a las excluidas 
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    cat_cols = [col for col in cat_cols if col not in exclude]

    # Generar dataframe con las columnas categóricas, cantidad y valores únicos
    resumen = pd.DataFrame({
        'columna': cat_cols,
        'valores_unicos': [df[col].nunique() for col in cat_cols],
        'valores': [df[col].unique().tolist() for col in cat_cols]
    }).sort_values(by='valores_unicos', ascending=False)

    return resumen


def verificar_columnas_iguales(df, col1, col2, ignore_case=True):
    """
    Devuelve las filas donde los valores de dos columnas son iguales.

    Args:
    df (pd.DataFrame): DataFrame de entrada.
    col1 (str): Nombre de la primera columna.
    col2 (str): Nombre de la segunda columna.
    ignore_case (bool): Si True, compara ignorando mayúsculas/minúsculas.

    Returns:
    dict: {
        'df_coincidencias': DataFrame con filas donde col1 == col2,
        'cantidad': número total de coincidencias,
        'valores_iguales': lista de valores únicos que coinciden
    }
    """
    if ignore_case:
        comp = df[col1].astype(str).str.upper() == df[col2].astype(str).str.upper()
    else:
        comp = df[col1] == df[col2]

    df_iguales = df[comp]
    cantidad = comp.sum()
    valores_iguales = df_iguales[col1].unique().tolist()

    return {
        'cantidad': cantidad,
        'valores_iguales': valores_iguales
    }
    


def check_nulls_nans(df):

    """
    Devuelve un DataFrame con el resumen de NaNs y ceros por columna.
    Incluye columnas categóricas y numéricas.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    total= len(df)
    print(f"Número total de filas en el DataFrame: {total}")

    resumen = pd.DataFrame(index=df.columns)

    # Cantidad y porcentaje de NaNs
    resumen["n_nans"] = df.isna().sum()
    resumen["%_nans"] = 100 * resumen["n_nans"] / total

    # Solo contar ceros en columnas numéricas
    resumen["n_ceros"] = 0
    resumen["%_ceros"] = 0.0

    columnas_numericas = df.select_dtypes(include='number').columns
    resumen.loc[columnas_numericas, "n_ceros"] = (df[columnas_numericas] == 0).sum()
    resumen.loc[columnas_numericas, "%_ceros"] = 100 * (df[columnas_numericas] == 0).sum() / total

    # Redondeo
    resumen["%_nans"] = resumen["%_nans"].round(2)
    resumen["%_ceros"] = resumen["%_ceros"].round(2)

    # Filtrar para mostrar solo columnas con algún nulo o cero
    resumen_filtrado = resumen[(resumen["n_nans"] > 0) | (resumen["n_ceros"] > 0)]

    return resumen_filtrado.sort_values(by=["n_nans", "n_ceros"], ascending=False)



def resumen_visitas_mensuales(df, fecha_col='fecha', cliente_col='cliente_id'):
    resumen = (
        df.groupby(fecha_col)[cliente_col]
          .agg(
              clientes_unicos='nunique',
              total_registros='count'
          )
          .reset_index()
    )

    # Detectar clientes duplicados por mes
    duplicados = (
        df.groupby([fecha_col, cliente_col])
          .size()
          .reset_index(name='conteo')
          .query('conteo > 1')
          .groupby(fecha_col)[cliente_col]
          .apply(list)
          .reset_index(name='clientes_duplicados')
    )

    # Unir info de duplicados al resumen
    resumen = resumen.merge(duplicados, on=fecha_col, how='left')
    resumen['clientes_duplicados'] = resumen['clientes_duplicados'].apply(lambda x: x if isinstance(x, list) else [])

    return resumen