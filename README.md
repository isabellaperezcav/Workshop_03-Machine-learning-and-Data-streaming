# Workshop 3: Machine Learning and Data Streaming


This repository contains the solution for **Workshop 3: Machine Learning and Data Streaming**, a project focused on building a regression model to predict happiness scores of different countries using data from five CSV files.

The task includes:

* Exploratory Data Analysis (EDA)
* ETL processes
* Regression model training
* Data streaming with Kafka
* Prediction and database storage
* Model performance evaluation

Tools used include Python, Jupyter Notebook, Scikit-learn, Apache Kafka, and PostgreSQL.

---

## Implementation Details

### EDA and ETL:

* CSV analysis (missing values, data types, distributions)
* Feature selection based on correlation and consistency
* Preprocessing: encoding, scaling, imputation

### Model Training:

* 80-20 split with Scikit-learn
* Evaluated models: Ridge, Random Forest, XGBoost, Lasso
* Model serialization with `.pkl`

### Data Streaming:

* Apache Kafka usage
* Producer sends transformed data
* Consumer receives and sends to the db

### Prediction and Storage:

* Predictions made in the consumer using the `.pkl` model
* Storage in PostgreSQL database

### Model Evaluation:

* Metrics: MAE, RMSE, R²

---

## Model Selection and Results

| Model         |    MAE   |   RMSE   |    R²     |
|---------------|----------|----------|-----------|
| Ridge         | 0.243188 | 0.315927 | 0.917945  |
| XGBoost       | 0.309364 | 0.396228 | 0.870931  |
| Random Forest | 0.350952 | 0.454441 | 0.830220  |
| lightgbm      | 0.409087 | 0.519349 | 0.778257  |
| Lasso         | 0.459663 | 0.585803 | 0.717880  |


**Selected Model**: *Ridge* for its performance and accuracy

---

## Model Usage

The trained model is serialized as `ridge_model.pkl` and used by the Kafka consumer to make real-time predictions. The results are stored in PostgreSQL in the `predicciones` table.

---

## Folder Structure

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
│   ├── ridge_model.pkl
│   ├── df_clean.csv
│   ├── feautures_selection.csv
├── pdf/
│   ├── Documentacion.pdf
│   ├── work03_ETL_isabellaperezcav.mp4
├── docker-compose.yml
├── README.md
├── requirements.txt
```

---

## How to Run the Project

### Prerequisites

* Docker installed ([Installation guide](https://docs.docker.com/get-docker/))
* Python 3 and dependencies installed:

```bash
pip install -r requirements.txt
```

### Steps

1. **Start Kafka and Zookeeper**:

```bash
docker-compose up
```

2. **Run the Kafka Producer**:

```bash
cd KafkaETL
python KafkaProducer.py
```

3. **Run the Kafka Consumer**:

```bash
cd KafkaETL
python KafkaConsumer.py
```

4. **View Results**:

* Check results in PostgreSQL or in the terminal
* Explore `notebooks/001_eda_modelTraining.ipynb`
* Review `pdf/work03_ETL_isabellaperezcav.mp4`

---

## Notes

* Make sure Docker is running before starting Kafka.
* Check topic names in `producer.py` and `consumer.py` (default: `happiness_topic`).
* You can query the database with tools like PostgreSQL -> pgAdmin4

---

## Technologies Used

* **Python**: Pipeline development
* **Jupyter Notebook**: EDA and training
* **Scikit-learn**: Modeling
* **Kafka**: Data streaming
* **PostgreSQL**: Storage
* **Docker**: Infrastructure
* **CSV files**: Data source

---

## Future Improvements

* More advanced feature engineering
* Hyperparameter tuning (grid search, random search)

---

## Contact

For questions or comments, contact **Isabella Perez** at \[[isabellaperezcav@gmail.com](mailto:isabellaperezcav@gmail.com)].

---
