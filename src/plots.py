import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo global
sns.set_theme(style="whitegrid", palette="Set2")

def plot_histogram(df, column, bins=30, title=None, log=False, kde=True):
    """
    Plots a histogram of a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
        bins (int): Number of bins for the histogram.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None: Displays the histogram plot.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], bins=bins, kde=kde)
    plt.title(title or f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    if log:
        plt.xscale('log')
    plt.show()

def plot_feature_distributions(df, bins=30, cols=2, clip_outliers=True,
                                        include_categorical=True, log=False):
    """
    Gráfica las distribuciones de variables numéricas y categóricas del dataset

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        bins (int): Cantidad de bins para histogramas.
        cols (int): Cantidad de columnas por fila en los subplots.
        clip_outliers (bool): Si True, recorta outliers extremos usando percentiles.
        include_categorical (bool): Si True, incluye variables categóricas.

    Returns:
        None
    """
    df_plot = df.copy()

    # Variables numéricas
    numeric_features = df_plot.select_dtypes(include='number').columns

    # Recorte de outliers si se desea
    if clip_outliers:
        for col in numeric_features:
            # Recortar valores extremos al percentil 1 y 99
             lower, upper = df_plot[col].quantile([0.01, 0.99])
             df_plot[col] = df_plot[col].clip(lower, upper)


    n_num = len(numeric_features)
    rows_num = (n_num + cols - 1) // cols
    fig_num, axes_num = plt.subplots(rows_num, cols, figsize=(cols * 5.5, rows_num * 4))
    axes_num = axes_num.flatten()

    for i, feature in enumerate(numeric_features):
        sns.histplot(data=df_plot, x=feature, bins=bins,
                     kde=True, element='step', ax=axes_num[i])
        axes_num[i].set_title(f"Distribución de {feature}")
        if log:
            axes_num[i].set_xscale('log')

    for j in range(i + 1, len(axes_num)):
        fig_num.delaxes(axes_num[j])

    fig_num.suptitle("Distribuciones de Variables Numéricas", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Variables categóricas
    if include_categorical:
        categorical_features = df_plot.select_dtypes(include='object').nunique()
        categorical_features = categorical_features.index.tolist()

        if categorical_features:
            n_cat = len(categorical_features)
            rows_cat = (n_cat + cols - 1) // cols
            fig_cat, axes_cat = plt.subplots(rows_cat, cols, figsize=(cols * 5.5, rows_cat * 4))
            axes_cat = axes_cat.flatten()

            for i, feature in enumerate(categorical_features):
                sns.countplot(data=df_plot, x=feature, ax=axes_cat[i])
                axes_cat[i].set_title(f"Distribución de {feature}")
                axes_cat[i].tick_params(axis='x', rotation=30)

            for j in range(i + 1, len(axes_cat)):
                fig_cat.delaxes(axes_cat[j])

            fig_cat.suptitle("Distribuciones de Variables Categóricas", fontsize=16)
            plt.tight_layout()
            plt.show()


def plot_countplot(df, category_col, hue=None):
    """
    Muestra la distribución de canales dentro de cada región.
    Es útil para justificar la imputación por moda regional.

    Args:
        df (pd.DataFrame): DataFrame de entrada
    """
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(
        data=df,
        x=category_col,
        hue=hue,
        order=sorted(df[category_col].unique())
    )
    plt.title(f"Distribución de {hue} por {category_col}")
    plt.xlabel(f"{category_col.capitalize()}")
    plt.ylabel("Cantidad de registros (log)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend(title=hue.upper(), bbox_to_anchor=(1.05, 1), loc="upper left")
    # Agregar el número de datos en cada barra
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black', rotation=0)
    plt.tight_layout()
    plt.show()


def plot_series_mensuales_subplots(df, date_col='fecha', agg='sum', exclude_cols=None):
    """
    Genera un gráfico con subplots para las series temporales mensuales 
    de todas las columnas numéricas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos.
        date_col (str): Nombre de la columna de fecha.
        agg (str): Método de agregación ('sum', 'mean', 'median', etc.).
        exclude_cols (list): Lista de columnas numéricas a excluir.

    Returns:
        None. Muestra el gráfico.
    """
    # Asegurar que la columna fecha esté en datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Excluir columnas no deseadas
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Crear figura con subplots
    n_cols = 2  # columnas en la cuadrícula
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4), sharex=True)

    axes = axes.flatten()

    # Graficar cada columna
    for i, col in enumerate(numeric_cols):
        serie = df.groupby(date_col)[col].agg(agg).sort_index()
        axes[i].plot(serie.index, serie.values, marker='o')
        axes[i].set_title(f'{agg.capitalize()} mensual de {col}')
        axes[i].grid(True)

    # Eliminar ejes vacíos si sobran
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_vs_target(df, target, features=None, hue=None, cols=2, figsize=(14, 5)):
    """
    Grafica subplots de todas las variables numéricas vs el target.
    """
    if features is None:
        features = [col for col in df.select_dtypes(include='number').columns if col != target]
    rows = (len(features) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(figsize[0], rows * figsize[1]))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        sns.scatterplot(x=feature, y=target, data=df, hue=hue, ax=axs[i])
        axs[i].set_title(f'{feature} vs {target}')
    # Oculta los ejes vacíos si hay
    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)
    plt.tight_layout()
    plt.show()

