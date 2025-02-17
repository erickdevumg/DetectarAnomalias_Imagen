# Usar
# python test_anomaly_detector.py --model anomaly_detector.model --image examples/highway_a836030.jpg

# importar los paquetes necesarios
from ia_umg.features import quantify_image
import argparse
import pickle
import cv2

# construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", required=True, help="path to trained anomaly detection model"
)
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# cargar el modelo de detección de anomalías
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())

# cargue la imagen de entrada, conviértala al espacio de color HSV y
# cuantificar la imagen de *la misma manera* como lo hicimos durante el entrenamiento
image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = quantify_image(hsv, bins=(3, 3, 3))

# utilizar el modelo de detector de anomalías y las características extraídas para determinar
# Si la imagen de ejemplo es una anomalía o no
preds = model.predict([features])[0]
label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)
print(label)

# dibujar el texto de la etiqueta previsto en la imagen original
cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# mostrar la imagen
cv2.imshow("Output", image)
cv2.waitKey(0)
