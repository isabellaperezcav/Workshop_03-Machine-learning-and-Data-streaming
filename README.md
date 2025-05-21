# Taller 3: Aprendizaje Automático y Transmisión de Datos

## Resumen

Este repositorio contiene la solución para el **Taller 3: Aprendizaje Automático y Transmisión de Datos**, un proyecto enfocado en construir un modelo de regresión para predecir puntajes de felicidad de diferentes países utilizando datos de cinco archivos CSV. 

La tarea incluye:

- Análisis Exploratorio de Datos (EDA)
- Procesos ETL
- Entrenamiento de modelo de regresión
- Transmisión de datos con Kafka
- Predicción y almacenamiento en base de datos
- Evaluación del rendimiento del modelo

Se usaron herramientas como Python, Jupyter Notebook, Scikit-learn, Apache Kafka y PostgreSQL


---

## Detalles de la Implementación

### EDA y ETL:

- Análisis de CSV (valores faltantes, tipos de datos, distribuciones)
- Selección de características por correlación y consistencia 
- Preprocesamiento: codificación, escalado, imputación

### Entrenamiento del Modelo:

- División 80-20 con Scikit-learn
- Modelos evaluados: Ridge, Random Forest, XGBoost, Lasso
- Serialización del modelo con `.pkl`

### Transmisión de Datos:

- Uso de Apache Kafka
- Productor envía datos transformados
- Consumidor los recibe y manda a la db

### Predicción y Almacenamiento:

- Predicciones realizadas en el consumidor con el modelo `.pkl`
- Almacenamiento en base de datos PostgreSQL

### Evaluación del Modelo:

- Métricas: MAE, RMSE, R²

---

## Selección del Modelo y Resultados

| Modelo        | MAE      | RMSE     | R²        | Notas                             |
|---------------|----------|----------|-----------|-----------------------------------|
| Ridge         | 0.003499 | 0.004596 | 0.999983  | Robusto, pero menos preciso       |
| Random Forest | 0.008105 | 0.018131 | 0.999730  | Mejor rendimiento general         |
| XGBoost       | 0.060757 | 0.083204 | 0.994309  | Buen rendimiento, más lento       |
| Lasso         | 0.080578 | 0.096873 | 0.992285  | Modelo de referencia simple       |

**Modelo Seleccionado**: *Random Forest* por su rendimiento y precisión

---

## Uso del Modelo

El modelo entrenado se serializa como `random_forest_model.pkl` y es utilizado por el consumidor Kafka para realizar predicciones en tiempo real. Los resultados se almacenan en PostgreSQL en la tabla `predicciones`.

---

## Estructura de Carpetas

```

Workshop\_03-Machine-learning-and-Data-streaming/
├── data/
│   ├── happiness\_2015.csv
│   ├── happiness\_2016.csv
│   ├── happiness\_2017.csv
│   ├── happiness\_2018.csv
│   ├── happiness\_2019.csv
│   ├── df_combined.csv
├── KafkaETL/
│   ├── KafkaProducer.py
│   ├── KafkaConsumer.py
├── notebooks/
│   ├── eda\_and\_model\_training.ipynb
├── models/
│   ├── random_forest_model.pkl
│   ├── df_clean.csv
│   ├── feautures_selection.csv
├── pdf/
│   ├── Documentacion.pdf
│   ├── work03_ETL_isabellaperezcav.mp4
├── docker-compose.yml
├── README.md
├── requirements.txt

````

---

## Cómo Ejecutar el Proyecto

### Requisitos Previos

- Docker instalado ([Guía de instalación](https://docs.docker.com/get-docker/))
- Python 3 y dependencias instaladas:

```bash
pip install -r requirements.txt
````

### Pasos

1. **Iniciar Kafka y Zookeeper**:

```bash
docker-compose up
```

2. **Ejecutar el Productor Kafka**:

```bash
cd KafkaETL
python KafkaProducer.py
```

3. **Ejecutar el Consumidor Kafka**:

```bash
cd KafkaETL
python KafkaConsumer.py
```

4. **Ver Resultados**:

* Consultar resultados en PostgreSQL o en la terminal 
* Explorar `notebooks/eda_and_model_training.ipynb`
* Revisar `pdf/demostracion.pdf`

---

## Notas

* Asegúrate de que Docker esté corriendo antes de iniciar Kafka.
* Revisa los nombres del topic en `producer.py` y `consumer.py` (por defecto: `happiness_data`).
* Puedes consultar la base de datos con herramientas como PostgreSQL -> pgAdmin4

---

## Tecnologías Utilizadas

* **Python**: Desarrollo del pipeline
* **Jupyter Notebook**: EDA y entrenamiento
* **Scikit-learn**: Modelado
* **Kafka**: Transmisión de datos
* **PostgreSQL**: Almacenamiento
* **Docker**: Infraestructura
* **Archivos CSV**: Fuente de datos

---

## Mejoras Futuras

* Ingeniería de características más avanzada
* Búsqueda de hiperparámetros (grid search, random search)

---

## Contacto

Para preguntas o comentarios, contacta a **Isabella Perez** en \[[isabellaperezcav@gmail.com](mailto:isabellaperezcav@gmail.com)].