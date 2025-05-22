import json
import os
import joblib
import numpy as np
import pandas as pd
import psycopg2
from kafka import KafkaConsumer
from dotenv import load_dotenv

load_dotenv("C:/Users/ASUS/Desktop/workshop3ETL/.env")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")


model = joblib.load(
    "C:/Users/ASUS/Desktop/workshop3ETL/models/ridge_model.pkl"
)

features_path = (
    "C:/Users/ASUS/Desktop/workshop3ETL/models/features_selection.csv"
)
selected_features = pd.read_csv(features_path)["feature"].tolist()


conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)
cur = conn.cursor()


boolean_features = [f for f in selected_features if f.startswith("country_")]
float_features = [f for f in selected_features if f not in boolean_features]

feature_columns_sql = ",\n".join(
    [f"{f} FLOAT" for f in float_features]
    + [f"{f} BOOLEAN" for f in boolean_features]
)

cur.execute(
    f"""
    CREATE TABLE IF NOT EXISTS predicciones (
        id SERIAL PRIMARY KEY,
        {feature_columns_sql},
        happiness_score_pred FLOAT,
        valor_real FLOAT,
        acierto BOOLEAN,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)
conn.commit()


consumer = KafkaConsumer(
    "happiness_topic",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)


def normalize_value(v):
    """Convierte tipos NumPy a nativos para psycopg2."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v  # bool, None, str, etc.


def process_message(message):
    try:
        data = message.value
        # Seleccionar features presentes en el mensaje
        input_data = {k: data[k] for k in selected_features if k in data}
        df = pd.DataFrame([input_data])

        if not set(selected_features).issubset(df.columns):
            print("[✘] Faltan columnas necesarias en los datos")
            return

        # Predicción
        prediction = float(model.predict(df[selected_features])[0])
        print(prediction)

        # Valor real y acierto
        try:
            valor_real = float(data.get("valor_real"))
            acierto = abs(valor_real - prediction) < 0.1
        except (TypeError, ValueError):
            valor_real, acierto = None, None

        # Preparar lista de valores en orden de columnas
        values_list = [normalize_value(data.get(f)) for f in selected_features]

        # INSERT dinámico
        placeholders = ", ".join(["%s"] * len(values_list))
        cols = ", ".join(selected_features)
        insert_query = f"""
            INSERT INTO predicciones ({cols}, happiness_score_pred, valor_real, acierto)
            VALUES ({placeholders}, %s, %s, %s)
        """
        cur.execute(insert_query, values_list + [prediction, valor_real, acierto])
        conn.commit()

        print(
            f"[✔] Predicción: {prediction:.4f} | Guardado en DB | Acierto: {acierto}"
        )
    except Exception as e:
        conn.rollback()
        print(f"[!] Error procesando mensaje: {e}")



print("Esperando mensajes de Kafka…")
for msg in consumer:
    process_message(msg)
