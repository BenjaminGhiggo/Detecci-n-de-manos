import cv2
import mediapipe as mp

# Inicializa MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Detección de Manos en Tiempo Real (Espejo)", cv2.WINDOW_NORMAL)

# Obtiene el tamaño de la pantalla
screen_width = cv2.getWindowImageRect("Detección de Manos en Tiempo Real (Espejo)")[2]
screen_height = cv2.getWindowImageRect("Detección de Manos en Tiempo Real (Espejo)")[3]

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.resizeWindow("Detección de Manos en Tiempo Real (Espejo)", screen_width, screen_height)
cv2.moveWindow("Detección de Manos en Tiempo Real (Espejo)", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Si se detectan manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos y conexiones de las manos
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Redimensiona el frame al tamaño de la ventana
    window_width = cv2.getWindowImageRect("Detección de Manos en Tiempo Real (Espejo)")[2]
    window_height = cv2.getWindowImageRect("Detección de Manos en Tiempo Real (Espejo)")[3]
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Detección de Manos en Tiempo Real (Espejo)", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
