import cv2
import numpy as np

# Inicializa el detector de características ORB
orb = cv2.ORB_create(nfeatures=500)

# Color para dibujar las características
FEATURE_COLOR = (0, 255, 0)  # Verde

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Reconocimiento de Características en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Reconocimiento de Características en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a escala de grises para ORB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta puntos clave y descriptores
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Dibuja los puntos clave en el frame
    frame = cv2.drawKeypoints(frame, keypoints, None, color=FEATURE_COLOR, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Reconocimiento de Características en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Reconocimiento de Características en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Reconocimiento de Características en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
