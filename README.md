# DetectarAnomalias_Imagen
1. Ejecutar el modelo a entrenar:
   python train_anomaly_detector.py --dataset forest --model anomaly_detector.model
2. Ejecutar la prueba:
   python test_anomaly_detector.py --model anomaly_detector.model --image examples/highway_a836030.jpg

   Cambiar la imagen para probar con otras, agregue nuevas imagenes para ver el compartamiento del modelo entrenado o si debe volverlo a ejecutar.

#Nota si es pura perdida, da errores con los anteriores debe instalar los paquetes que solicite con pip install.
