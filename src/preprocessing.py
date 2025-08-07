import pandas as pd
import numpy as np

def estandarizar_categorias(df, columna, mapeo):
    """
    Mapeo y estandarización a una columna categórica.

    Args:
    df (pd.DataFrame): DataFrame de entrada.
    columna (str): Nombre de la columna a limpiar.
    mapeo (dict): Diccionario {valor_incorrecto: valor_correcto}.

    Returns:
    pd.DataFrame: DataFrame con columna corregida.
    """
    df_estandarizada = df.copy()
    col = df_estandarizada[columna].astype(str)
    col = col.str.upper()
    df_estandarizada[columna] = col.map(mapeo).fillna(col)  
    return df_estandarizada


def imputar_por_cliente(df, col_imp, col2):
    """
    Imputa valores erróneos de canal usando el historial del mismo cliente.
    
    Esta función corrige registros donde canal == region buscando el canal 
    correcto en otros períodos del mismo cliente_id.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas:
            - cliente_id: Identificador único del cliente
            - canal: Canal del cliente (puede contener errores)
            - region: Región del cliente
            - Otras columnas se mantienen sin cambios
    
    Returns:
        pandas.DataFrame: DataFrame corregido donde:
            - Se mantienen todas las columnas originales
            - Los valores de 'canal' problemáticos (donde canal == region) 
              se reemplazan por el canal más frecuente del mismo cliente_id
            - Si un cliente no tiene registros válidos, mantiene el valor original
    """

    df_clean = df.copy()
    
    # Identificar registros problemáticos
    mask_problematicos = df_clean[col_imp] == df_clean[col2]
    
    # Para cada cliente problemático, buscar su canal en otros períodos
    for idx in df_clean[mask_problematicos].index:
        cliente_id = df_clean.loc[idx, 'cliente_id']
        
        # Buscar otros registros del mismo cliente con canal válido
        otros_registros = df_clean[
            (df_clean['cliente_id'] == cliente_id) & 
            (df_clean[col_imp] != df_clean[col2])
        ]
        
        if not otros_registros.empty:
            # Tomar el canal más frecuente para este cliente
            canal_moda = otros_registros[col_imp].mode()
            if len(canal_moda) > 0:
                df_clean.loc[idx, col_imp] = canal_moda.iloc[0]
    
    return df_clean


def imputar_por_moda_regional(df):
    df_clean = df.copy()
    
    # Detectar valores mal cargados
    mask_problematicos = df_clean['canal'] == df_clean['region']
    
    # Obtener moda de canal por región excluyendo los problemáticos
    df_validos = df_clean[~mask_problematicos]
    moda_por_region = df_validos.groupby('region')['canal'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN').to_dict()
    
    # Imputar usando map vectorizado
    df_clean.loc[mask_problematicos, 'canal'] = df_clean.loc[mask_problematicos, 'region'].map(moda_por_region)
    
    return df_clean

def replace_placeholders(df, placeholders=[-999999, 999999]):
    """
    Reemplaza los valores placeholders por NaN en todas las columnas del DataFrame.

    Args:
    df (pd.DataFrame): DataFrame de entrada.
    placeholders (list): Lista de valores a reemplazar por NaN.

    Returns:
    pd.DataFrame: DataFrame con placeholders reemplazados.
    """
    df_reemplazado = df.copy()
    # Reemplazar placeholders por NaN
    df_reemplazado = df_reemplazado.replace(placeholders, np.nan)
    return df_reemplazado

def eliminar_negativos(df, columnas):
    """
    Reemplaza valores negativos por NaN en las columnas especificadas.

    Args:
    df (pd.DataFrame): DataFrame.
    columnas (list): Lista de nombres de columnas a procesar.

    Returns:
    pd.DataFrame: DataFrame con valores negativos reemplazados por NaN.
    """
    df_sin_negativos = df.copy()
    for col in columnas:
        if col in df_sin_negativos.columns:
            df_sin_negativos.loc[df_sin_negativos[col] < 0, col] = np.nan
    return df_sin_negativos


def preprocesamiento_pipeline(df, target, cols_numericas):
    
    """
    Pipeline de preprocesamiento de datos.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target (str): Nombre de la columna objetivo.
        cols_numericas (list): Lista de columnas numéricas a procesar.   
    Returns:
        pd.DataFrame: DataFrame preprocesado.   
    
    """

    # Drop duplicados
    df = df.drop_duplicates()

    # Estandarizar variables categóricas
    mapeo_canal = {
    'AUTOSERVICIO': 'AUTOSERVICIO',
    'AUTOSERVICIOS': 'AUTOSERVICIO',
    'KIOSCOS/MAXIKIOSCOS': 'KIOSCOS/MAXIKIOSCOS',
    'KIOS/MAXIKIOSCKO': 'KIOSCOS/MAXIKIOSCOS',
    'GBA MINO': 'GBA MINORISTAS',
    'CENTRO': 'CENTRAL'
    }
    df = estandarizar_categorias(df, 'canal', mapeo=mapeo_canal)
    
    # Imputar valores erróneos en 'canal' por cliente y región
    df = imputar_por_cliente(df, 'canal', 'region')
    df = imputar_por_moda_regional(df)
    
    # Reemplazar placeholders por NaN
    df = replace_placeholders(df)
    
    # Eliminar filas con NaN en el target
    df = df.dropna(subset=[target])
    
    # Imputar valores negativos y crear columnas indicadoras
    for col in cols_numericas:
        df[col + '_negativa'] = df[col] < 0
        mediana = df.loc[df[col] >= 0, col].median()
        df.loc[df[col] < 0, col] = mediana
    
    # Imputar NaN en ventas como ceros
    df = df.fillna(0)
    
    return df