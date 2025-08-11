from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analisis_completo_columnas(df):
    resumen = []
    
    for col in df.columns:
        # Estadísticas básicas
        media = df[col].mean()
        mediana = df[col].median()
        std = df[col].std()
        
        # Forma de la distribución
        asimetria = df[col].skew()
        curtosis = df[col].kurtosis()
        
        # Outliers IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = len(df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)])
        
        # Test de normalidad
        _, p_value_shapiro = stats.shapiro(df[col].sample(min(5000, len(df))))
        es_normal = p_value_shapiro > 0.05
        
        resumen.append({
            'columna': col,
            'media': media,
            'mediana': mediana,
            'std': std,
            'asimetria': asimetria,   # 0: normal, [-0.5,0.5] aprox normal, >1 sesgada a la der, <-1 sesgada a la izq
            'curtosis': curtosis,  # 0: igual de puntiaguda q la normal, >0 mas mas puntiaguda, <0 menos 
            'outliers_iqr': outliers_iqr,
            'outliers_pct': outliers_iqr/len(df)*100,
            'es_normal': es_normal, # p-value test shapiro (probabilidad de que los datos provengan de una distribución normal) >0.05 normalidad
            'diferencia_media_mediana': abs(media - mediana)
        })
  
    return pd.DataFrame(resumen)


def high_correlation(df):
    plt.figure(figsize=(8,6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlaciones')
    plt.show()

    # Identificar columnas altamente correlacionadas
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    correlation_matrix.columns[i], 
                    correlation_matrix.columns[j], 
                    correlation_matrix.iloc[i, j]
                ))

    print("Pares de variables altamente correlacionadas:")
    for pair in high_corr_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")


def agrupar_onehot_importance(feat_importance_df, prefijos_categoricos):
    df = feat_importance_df.copy()

    grupos = []
    df_restante = df.copy()

    for prefijo in prefijos_categoricos:
        mask = df_restante['feature'].str.startswith(prefijo)
        suma = df_restante.loc[mask, 'importance'].sum()
        nombre_grupo = prefijo.rstrip('_')  
        grupos.append({'feature': nombre_grupo, 'importance': suma})
        # Eliminar esas filas para no duplicar
        df_restante = df_restante.loc[~mask]

    # Crear df con grupos sumados
    df_grupos = pd.DataFrame(grupos)

    # Concatenar los que quedaron (no categóricos) con los grupos sumados
    df_final = pd.concat([df_restante, df_grupos], ignore_index=True)

    # Ordenar por importancia descendente
    df_final = df_final.sort_values('importance', ascending=False).reset_index(drop=True)

    return df_final



def error_ponderado_negocio(y_true, y_pred, pesos):
    errores = np.abs(y_true - y_pred)
    error_ponderado = np.sum(pesos * errores) / np.sum(pesos)
    return error_ponderado