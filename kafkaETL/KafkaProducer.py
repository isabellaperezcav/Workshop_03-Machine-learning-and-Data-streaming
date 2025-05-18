from kafka import KafkaProducer
import pandas as pd
import json
import time


ruta_features      = r"C:/Users/ASUS/Desktop/workshop3ETL/models/features_selection.csv"
ruta_datos_limpios = r"C:/Users/ASUS/Desktop/workshop3ETL/models/df_clean.csv"
ruta_datos_completos = r"C:/Users/ASUS/Desktop/workshop3ETL/data/df_combined.csv"


features = pd.read_csv(ruta_features)["Feature"].tolist()

df_feat   = pd.read_csv(ruta_datos_limpios)    
df_score  = pd.read_csv(ruta_datos_completos)  # contiene Happiness_Score para el valor real


assert len(df_feat) == len(df_score), "Los CSV no tienen el mismo número de filas"

# Filtrar variables de entrada
df_feat = df_feat[features]


producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


for idx, feat_row in df_feat.iterrows():
    mensaje = feat_row.to_dict()

    # Añadir el valor real desde el otro CSV
    valor_real = df_score.loc[idx, "Happiness_Score"]
    mensaje["valor_real"] = float(valor_real)        # asegurar que sea numérico

    producer.send("happiness_topic", value=mensaje)
    print(f"Mensaje enviado: {mensaje}")
    time.sleep(1)  # 1 s entre envios

producer.flush()
