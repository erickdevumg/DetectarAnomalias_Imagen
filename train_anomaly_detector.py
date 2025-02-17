# Usar
# python train_anomaly_detector.py --dataset forest --model anomaly_detector.model

# importar los paquetes necsarios
from ia_umg.features import load_dataset
from sklearn.ensemble import IsolationForest
import argparse
import pickle

# Construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset of images")
ap.add_argument(
    "-m", "--model", required=True, help="path to output anomaly detection model"
)
args = vars(ap.parse_args())

# cargar y cuantificar nuestro conjunto de datos de imágenes (dataset)
print("[INFO] preparing dataset...")
data, failed = load_dataset(args["dataset"], bins=(3, 3, 3))
print(f"[INFO] Número de imágenes cargadas: {len(data)}")
print(f"[INFO] Número de imágenes fallidas: {len(failed)}")

# guardar lista de rutas con intentos fallidos
with open(r"failed_paths.txt", "w") as fp:
    fp.write("\n".join(failed))

# entrenar el modelo de detección de anomalías
print("[INFO] fitting anomaly detection model...")
if len(data) < 20:  # Si hay muy pocas imágenes, reducimos los estimadores
    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
else:
    model = IsolationForest(
        n_estimators=1000,
        contamination="auto",
        n_jobs=5,
        max_features=3,
        random_state=42,
    )
model.fit(data)

# serializar el modelo de detección de anomalías en el disco
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
