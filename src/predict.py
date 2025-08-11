import pandas as pd
import numpy as np
from src.feature_engineering import features_eng

def predict_from_csv(filepath, model):
    """
    Predice los minutos de visita por cliente.
    El archivo de entrada debe contener las mismas características que se usaron en el entrenamiento.

    Args:
        filepath (str): Ruta al archivo CSV con los nuevos datos (no se requiere la columna 'minutos_visitados_mes').
        model (object): Modelo Catboost entrenado.

    Returns:
        pd.DataFrame: Datos de entrada con una columna adicional para los valores predichos.
    """
    # Load the new data
    df_new = pd.read_csv(filepath)

    # Apply same transformations as in training
    output = features_eng(df_new)
    
    #predict log-salaries and revert the log transform
    y_pred= model.predict(output)


    # Add predictions to the original DataFrame
    df_new = df_new.copy()
    df_new['minutos_visita_predicción'] = y_pred
    df_new.to_csv('predicciones.csv', index=False)


    return df_new

import joblib

file_path = '../data/nuevos_datos.csv'
model = joblib.load('../models/modelo_catboost.pkl')
predict_from_csv(file_path, model)