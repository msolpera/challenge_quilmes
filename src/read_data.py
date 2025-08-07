from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/Dataset.csv"

def load_dataset() -> pd.DataFrame:
    """Load dataset"""
    df = pd.read_csv(DATA_PATH, delimiter=',')

    # Crear columna 'fecha'
    df['fecha'] = pd.to_datetime(df['aniomes'], format='%Y%m')

    # Reordenar columnas para que 'fecha' esté después de 'aniomes'
    cols = list(df.columns)
    cols.insert(cols.index('aniomes') + 1, cols.pop(cols.index('fecha')))
    df = df[cols]

    return df

