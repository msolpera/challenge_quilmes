# Proyecto Data Science - Recomendación de Tiempo de Visita para Clientes

## Contexto

La empresa se dedica a la venta de bienes de consumo masivo en un entorno B2B, con una extensa red de vendedores que visitan a los clientes presencialmente. El equipo de Nivel de Servicio está enfocado en mejorar la experiencia del cliente durante estas visitas offline, optimizando la calidad y duración de las mismas.

Este proyecto tiene como objetivo desarrollar una herramienta basada en Machine Learning que recomiende el tiempo óptimo de visita para cada cliente, a partir de variables comerciales y de comportamiento.

---

## Descripción del Proyecto

Se busca calcular la duración total recomendada de visitas mensuales (en minutos) para cada cliente, utilizando datos históricos y características comerciales. Además, se genera un score normalizado de recomendación dentro de cada región, con el fin de obtener un ranking regional que facilite la priorización y gestión de las visitas.

---

## Datos

El dataset suministrado contiene la siguiente información por cliente y mes:

- `cliente_id`: Código único del cliente.
- `aniomes`: Periodo año-mes del registro (formato `yyyyMM`).
- `canal`: Canal al que pertenece el cliente (por ejemplo, Autoservicio, Kiosco Maxiquiosco).
- `region`: Región geográfica del cliente.
- `minutos_visitados_mes`: Minutos totales que se visitó el cliente en el mes.
- `cantidad_heladeras`: Número de heladeras en el cliente durante el periodo.
- `venta_total_negocios_mes`: Venta total en unidades Hl del cliente en el mes.
- `cantidad_productos_total_negocios_vendidos_mes`: Cantidad de productos distintos vendidos en el mes.
- `cantidad_compras_total_negocios_mes`: Número total de compras realizadas por el cliente en el mes.
- `dias_entre_compras_total_negocios_mes`: Días promedio entre compras en el mes.
- `venta_negocio1_mes` a `venta_negocio4_mes`: Ventas en unidades Hl para cada uno de los 4 negocios.

---

## Análisis Exploratorio (EDA)

Se realiza un análisis para responder:

- ¿Cuántos clientes hay por región y canal?
- ¿Cuál es el ranking de clientes según ventas totales?
- ¿Cuál negocio es el que más ventas genera dentro de la compañía?
- ¿Existen valores duplicados o inconsistencias?
- ¿Hay valores atípicos (outliers) relevantes?

---

## Modelado

Se entrena un modelo de Machine Learning para predecir la cantidad de minutos recomendados para visitar cada cliente en el mes más reciente disponible, usando variables comerciales y del histórico de visitas.

---

## Score y Ranking

- Se calcula un **score normalizado** para la recomendación de minutos dentro de cada región, usando min-max scaling.
- Se genera un **ranking regional** para ordenar a los clientes en cada región según el score obtenido.

---

## Resultados

- Tabla con predicción de minutos recomendados, score normalizado y ranking regional para cada cliente.
- Análisis interpretativo para el equipo de Nivel de Servicio explicando la utilidad de la herramienta y los insights obtenidos (Power Bi y pdf).
  

---

## Uso

Cargar los datos en carpeta `data`  y correr el archivo main.ipynb.


---

## Estructura del repo

data/ # Carpeta con datos de entrada y salida

    ├── predictions/ # Resultados de predicciones generadas
    │ └── predictions.csv # Archivo CSV con las predicciones
    ├── processed/ # Datos procesados intermedios
    │ ├── df_clean.csv # Dataset limpio
    │ ├── df_feat.csv # Dataset con features generados
    ├── Dataset.csv # Dataset original crudo
    models/ # Modelos entrenados guardados
    ├── modelo_catboost.pkl # Modelo CatBoost 
    ├── modelo_linear.pkl # Modelo lineal 
    notebooks/ # Jupyter notebooks para exploración y modelado
    ├── catboost_info/ # Notebooks o recursos específicos de CatBoost
    ├── EDA.ipynb # Notebook de Análisis Exploratorio de Datos
    └── modelling.ipynb # Notebook para entrenamiento y evaluación de modelos
    src/ # Código fuente
    ├── pycache/ # Cache de Python (archivos compilados .pyc)
    ├── clean.py # Script para limpieza de datos
    ├── eda.py # Script para análisis exploratorio
    ├── feature_engineering.py# Script para creación de variables (features)
    ├── model.py # Script para definir y entrenar modelos
    ├── plots.py # Funciones para gráficos y visualizaciones
    ├── preprocesing.py # Script para preprocesamiento de datos 
    ├── read_data.py # Funciones para carga de datos
    └── utils.py # Funciones utilitarias varias

## Tecnologías y Librerías

- Python 3.8 o superior

- CatBoost

- NumPy

- Pandas

- Scikit-learn

- Matplotlib

- Seaborn

- Scipy

---

## Consideraciones y Supuestos

- En caso de datos faltantes o inconsistencias, se aplicaron estrategias de imputación.
- Se asumió que el score normalizado permite comparabilidad justa dentro de regiones.
- El modelo está orientado a optimizar tiempos de visita, buscando eficiencia en la red de vendedores.

## Instalación

1. Clonar el repositorio

    git clone https://github.com/msolpera/challenge_quilmes

    cd challenge_quilmes

2. Crear entorno

python -m venv env

source env/bin/activate      # Linux / Mac

env\Scripts\activate         # Windows

3. Instalar librerías

pip install -r requirements.txt
---

### Resultados
El modelo logra reducir el MAE en aproximadamente un 50% respecto al baseline y presenta un R² entre 0.58 y 0.68 en los conjuntos de validación y test.

## Contacto
María Sol Pera
msolpera@gmail.com

---





