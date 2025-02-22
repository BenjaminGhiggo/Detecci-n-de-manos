import pyttsx3
import cv2
import numpy as np

# Inicializa pyttsx3 para Text to Speech
engine = pyttsx3.init()

# Configura la velocidad y el volumen de la voz
engine.setProperty('rate', 150)    # Velocidad de habla (más bajo es más lento)
engine.setProperty('volume', 1)    # Volumen (1 es el máximo)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Text to Speech en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Text to Speech en Tiempo Real", 0, 0)

# Texto inicial
text = "Escribe un texto y presiona Enter para escuchar"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Crea una imagen negra para el cuadro de texto
    overlay = np.zeros(frame.shape, dtype=np.uint8)

    # Escribe el texto en la imagen
    cv2.putText(overlay, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Combina el cuadro de texto con el frame de la cámara
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Text to Speech en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Text to Speech en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Text to Speech en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ingresa texto y convierte a voz
    if cv2.waitKey(1) & 0xFF == ord('\r'):  # Enter
        engine.say(text)
        engine.runAndWait()
        text = input("Escribe un texto para convertir a voz: ")

# Limpieza
cap.release()
cv2.destroyAllWindows()
