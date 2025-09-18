# Predicción de la Calidad del Agua Dulce con Redes Neuronales

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Librerías-Scikit--learn%20%7C%20Optuna%20%7C%20Pandas-orange.svg)

Este proyecto utiliza una Red Neuronal Artificial (RNA) para resolver un problema de clasificación del mundo real: determinar si una muestra de agua dulce es apta para el consumo. El proceso sigue la metodología CRISP-DM y aborda desafíos comunes como datos faltantes, variables categóricas y desbalanceo de clases.

---

## 📚 Tabla de Contenidos
1.  [Fase 1: Comprensión del Negocio y los Datos](#fase-1-comprensión-del-negocio-y-los-datos)
2.  [Fase 2: Análisis Exploratorio de Datos (AED)](#fase-2-análisis-exploratorio-de-datos-aed)
3.  [Fase 3: Preparación de los Datos](#fase-3-preparación-de-los-datos)
4.  [Fase 4: Construcción del Modelo](#fase-4-construcción-del-modelo)
5.  [Fase 5: Evaluación y Conclusiones](#fase-5-evaluación-y-conclusiones)
6.  [Cómo Ejecutar este Proyecto](#cómo-ejecutar-este-proyecto)

---

## Fase 1: Comprensión del Negocio y los Datos

### 1.1. Comprensión del Negocio

#### El Problema Global: La Escasez de Agua Potable
El agua dulce es uno de los recursos naturales más vitales y escasos de nuestro planeta. Garantizar el acceso a un suministro de agua seguro es crucial para la salud pública y la supervivencia de los ecosistemas.

#### El Problema Operativo: Monitoreo Lento y Reactivo
Tradicionalmente, la evaluación de la calidad del agua se realiza mediante análisis de laboratorio, un proceso costoso, lento y reactivo. Existe una necesidad crítica de desarrollar métodos más rápidos y proactivos para evaluar la potabilidad del agua.

#### Objetivos del Proyecto y Criterios de Éxito
El objetivo principal es desarrollar un **modelo de Machine Learning de alto rendimiento** capaz de predecir si una muestra de agua es "Sostenible" (apta) o "No Sostenible" (no apta).

El éxito del proyecto se define por la capacidad del modelo para:
1.  **Clasificar correctamente** las muestras con una alta precisión general.
2.  **Minimizar los Falsos Negativos:** Es de vital importancia identificar correctamente el agua "No Sostenible". Este es nuestro criterio de éxito principal.

### 1.2. Comprensión de los Datos

El proyecto utiliza un conjunto de datos público de Kaggle, originado en el "Intel OneAPI Online AI Hackathon".

- **Dataset en Kaggle:** [Predict the Quality of Freshwater](https://www.kaggle.com/datasets/naiborhujosua/predict-the-quality-of-freshwater)

El dataset consta de **~5.9 millones de registros** y 22 características. Cumple con todos los requisitos del proyecto:
- ✅ **Datos Faltantes:** Múltiples columnas presentan valores nulos.
- ✅ **Variables Categóricas:** Incluye `Color`, `Source` y `Month`.
- ✅ **Desbalanceo de Clases:** La clase "No Sostenible" es la mayoritaria.

---

## Fase 2: Análisis Exploratorio de Datos (AED)

En esta fase, profundizamos en los datos para descubrir patrones y tomar decisiones informadas para el preprocesamiento.

### 2.1. Desbalance de Clases
El análisis de la variable objetivo revela un significativo desbalance: el **70% de las muestras son "No Sostenibles" (Clase 0)**. Esto nos obliga a usar métricas como el F1-Score y el Recall, y a aplicar técnicas de balanceo en el modelado.

![Distribución de Clases](images/distribucion_clases.png)

### 2.2. Valores Faltantes
El dataset presenta una cantidad considerable de valores nulos. Sin embargo, ninguna característica supera el 4% de datos faltantes, lo que hace que las técnicas de imputación sean una estrategia viable y preferible a la eliminación masiva de filas.

![Valores Faltantes por Característica](images/valores_faltantes.png)

### 2.3. Distribuciones y Valores Atípicos (Outliers)
Los gráficos de violín revelan que, para contaminantes clave como la **Turbidez** y el **Hierro**, la **Clase 0 (No Sostenible)** presenta valores atípicos mucho más extremos. Esto valida que niveles altos de estos parámetros son indicadores de mala calidad y serán predictores potentes.

![Análisis de Outliers](images/analisis_outliers.png)

### 2.4. Análisis de Correlación
El mapa de calor confirma que las relaciones entre las características y la calidad del agua son complejas y **no lineales**, ya que la mayoría de las correlaciones directas son muy bajas. Esto justifica plenamente la elección de una Red Neuronal Artificial, un modelo capaz de capturar estos patrones complejos.

![Mapa de Correlación](images/mapa_correlacion.png)

---

## Fase 3: Preparación de los Datos

El preprocesamiento fue un paso crítico para preparar los datos para la Red Neuronal. Las principales acciones fueron:
- **Limpieza:** Se eliminaron las columnas irrelevantes como `Index`.
- **Codificación:** Las variables categóricas (`Color`, `Month`, etc.) se convirtieron a formato numérico usando `LabelEncoder`.
- **Imputación:** Se utilizó una estrategia de imputación robusta (`IterativeImputer`) para rellenar los valores faltantes.
- **División Estratificada:** Los datos se dividieron en conjuntos de entrenamiento (70%) y prueba (30%) de forma estratificada para mantener la proporción de clases.
- **Escalado:** Se aplicó `MinMaxScaler` a las características numéricas para normalizar sus rangos entre 0 y 1, un paso fundamental para el entrenamiento estable de la RNA.

---

## Fase 4: Construcción del Modelo

### 4.1. Diseño de la Arquitectura
La arquitectura de la Red Neuronal (MLP) se diseñó de forma justificada, con una capa de entrada correspondiente a las características, capas ocultas con activación **ReLU** para aprender patrones complejos, y una capa de salida con activación **Sigmoide** para la clasificación binaria. El optimizador elegido fue **Adam**.

### 4.2. Optimización de Hiperparámetros con Optuna
Para encontrar la mejor configuración de la red (número de capas, neuronas, tasa de aprendizaje), se utilizó **Optuna**, una librería de optimización automática. Se ejecutaron múltiples experimentos para encontrar la arquitectura que **maximizara el F1-Score de la Clase 0 (No Apta)**, alineando la optimización técnica con el objetivo de negocio.

---

## Fase 5: Evaluación y Conclusiones

Se entrenaron y evaluaron tres modelos para una comparación exhaustiva:
1.  **Modelo 1:** Datos SIN Normalizar (Baseline)
2.  **Modelo 2:** Datos Normalizados
3.  **Modelo 3:** Datos Normalizados + Ponderación de Clases

### 5.1. Tabla Comparativa de Resultados

| Métrica Clave | Modelo 1 (Sin Normalizar) | Modelo 2 (Normalizado) | Modelo 3 (Normalizado + Ponderado) |
| :--- | :---: | :---: | :---: |
| **Recall (Clase 0 - No Apta)** | 0.83 | 0.83 | **0.84** |
| **F1-Score (Clase 0 - No Apta)** | 0.90 | 0.90 | 0.90 |
| **Falsos Negativos (Riesgo)** | 206,530 | 206,272 | **204,553** |
| **Accuracy General** | 87% | 87% | 87% |
| **AUC** | 0.92 | 0.92 | 0.91 |
| **Estabilidad del Entrenamiento**| Inestable | **Estable** | **Estable** |

### 5.2. Conclusión Final

El análisis comparativo demuestra dos puntos clave:
1.  **La normalización es fundamental:** El modelo entrenado con datos normalizados (Modelo 2) mostró una convergencia mucho más estable que el modelo sin normalizar.
2.  **La ponderación de clases es efectiva:** El Modelo 3 logró el objetivo de negocio principal al obtener el **Recall más alto para la Clase 0 (0.84)** y, en consecuencia, **el menor número de Falsos Negativos (204,553)**.

**Modelo Ganador:** Se selecciona el **Modelo 3 (MLP Normalizado con Ponderación de Clases)** como la solución final. Aunque su rendimiento general es similar al de los otros, es el modelo más seguro y alineado con nuestro criterio de éxito principal.

![Matriz de Confusión del Modelo Ganador](images/matriz_confusion_final.png)

El modelo final es una herramienta robusta y prometedora, con una **precisión del 87%** y un **AUC de 0.91**, aunque para una implementación en el mundo real, se buscaría reducir aún más el 16% de Falsos Negativos restantes.

---

## Cómo Ejecutar este Proyecto

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```
2.  **Crear y Activar un Entorno Virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instalar las Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configurar la API de Kaggle:**
    - Descarga tu archivo `kaggle.json` desde tu cuenta de Kaggle.
    - Colócalo en la ruta `~/.kaggle/` (macOS/Linux) or `C:\Users\<TuUsuario>\.kaggle\` (Windows).

5.  **Ejecutar Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
