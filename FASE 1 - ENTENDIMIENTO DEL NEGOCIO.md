# Proyecto: Predicción de la Calidad del Agua Dulce

## 1. Fase 1: Comprensión del Negocio y los Datos

En esta fase inicial, establecemos el contexto, los objetivos y el plan del proyecto. Definimos el problema desde una perspectiva global y de negocio, lo traducimos a un problema de ciencia de datos y realizamos un primer análisis para comprender el material con el que trabajaremos: los datos.

### 1.1. Comprensión del Negocio

#### El Problema Global: La Escasez de Agua Potable

El agua dulce es uno de los recursos naturales más vitales y escasos de nuestro planeta, representando apenas el 3% del volumen total de agua. Es fundamental para casi todos los aspectos de la vida humana, desde el consumo directo y la higiene hasta la generación de alimentos y electricidad. Sin embargo, este recurso esencial se encuentra bajo una presión cada vez mayor debido a la sequía, la contaminación y el aumento de las temperaturas globales. Garantizar el acceso a un suministro de agua seguro no solo es crucial para la salud pública, sino también para la supervivencia de los ecosistemas que dependen de ella.

#### El Problema Operativo: Monitoreo Lento y Reactivo

Tradicionalmente, la evaluación de la calidad del agua se realiza mediante análisis fisicoquímicos en laboratorios. Este proceso, aunque preciso, es costoso, requiere mucho tiempo y es inherentemente reactivo. Una muestra puede tardar días o semanas en ser analizada, tiempo durante el cual una fuente de agua contaminada podría ya haber afectado a una comunidad o a un ecosistema. Existe una necesidad crítica de desarrollar métodos más rápidos, escalables y proactivos para evaluar la potabilidad del agua en tiempo casi real.

#### Objetivos del Proyecto y Criterios de Éxito

El objetivo principal de este proyecto es desarrollar un **modelo de Machine Learning de alto rendimiento** capaz de predecir si una muestra de agua dulce es apta para el consumo ("Sostenible") basándose en sus parámetros medibles.

El éxito del proyecto se definirá por la capacidad del modelo para:
1.  **Clasificar correctamente** las muestras de agua con una alta precisión general.
2.  **Minimizar los Falsos Negativos:** Es de vital importancia identificar correctamente el agua "No Sostenible". Clasificar erróneamente una muestra peligrosa como segura tiene consecuencias mucho más graves que el error inverso.
3.  **Ser Eficiente:** El modelo debe ser capaz de procesar grandes volúmenes de datos y ofrecer predicciones rápidas, superando las limitaciones de los métodos de laboratorio tradicionales.

### 1.2. Comprensión de los Datos

El proyecto utiliza un conjunto de datos público de Kaggle, originado en el "Intel OneAPI Online AI Hackathon".

- **Dataset en Kaggle:** [Predict the Quality of Freshwater](https://www.kaggle.com/datasets/naiborhujosua/predict-the-quality-of-freshwater)

#### Descripción General del Dataset

El dataset consta de **5,956,842 registros** de muestras de agua, cada uno con 22 características (parámetros fisicoquímicos y contextuales) y una etiqueta objetivo.

#### La Variable Objetivo (`Target`)

La tarea es un **problema de clasificación binaria desbalanceada**. El objetivo es predecir la columna `Target`, que indica la sostenibilidad del agua.

| Sostenibilidad del Agua | Etiqueta | No. de Muestras | Porcentaje |
| :--- | :---: | :---: | :---: |
| No Sostenible | `Target: 0` | 4,151,590 | **69.69%** |
| Sostenible | `Target: 1` | 1,805,252 | **30.31%** |

El **desbalance de clases** es un desafío clave: la clase "No Sostenible" es más del doble de frecuente que la clase "Sostenible". Esto debe ser abordado en las fases de preprocesamiento y modelado para evitar que el modelo se vuelva sesgado hacia la clase mayoritaria.

#### Descripción de las Características (`Features`)

Los parámetros medidos en cada muestra de agua se pueden agrupar en las siguientes categorías:

| Categoría | Características Incluidas |
| :--- | :--- |
| **Parámetros Fisicoquímicos** | `pH`, `Turbidity`, `Odor`, `Conductivity`, `Total dissolved solids`, `Color` |
| **Contenido Mineral y Químico**| `Iron`, `Nitrate`, `Chloride`, `Lead`, `Zinc`, `Fluoride`, `Copper`, `Sulfate`, `Chlorine`, `Manganese` |
| **Datos Contextuales** | `Source`, `Water temperature`, `Air temperature`, `Month`, `Day`, `Time of day` |

#### Desafíos Iniciales Identificados en los Datos

Un análisis preliminar revela varios desafíos técnicos que guiarán nuestro proceso de preprocesamiento y modelado:

1.  **Volumen Masivo de Datos:** Con casi 6 millones de filas, la eficiencia del código y el manejo de la memoria son cruciales.
2.  **Gran Cantidad de Valores Faltantes:** El informe de inspiración señala que aproximadamente 2 millones de filas contienen al menos un valor faltante. Se requerirá una estrategia de imputación o eliminación robusta.
3.  **Baja Correlación Lineal:** Las correlaciones directas entre las características y la variable objetivo son bajas. Esto sugiere que las relaciones son no lineales y complejas, justificando el uso de modelos avanzados como las Redes Neuronales en lugar de modelos lineales simples.
4.  **Dimensionalidad y Ruido:** La alta cantidad de características puede introducir ruido y aumentar la complejidad computacional, haciendo necesaria una selección o ingeniería de características cuidadosa.
