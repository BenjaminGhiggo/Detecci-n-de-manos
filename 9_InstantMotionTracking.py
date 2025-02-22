import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Holistic y Drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Instant Motion Tracking en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Instant Motion Tracking en Tiempo Real", 0, 0)

# Almacena las posiciones anteriores de los landmarks
prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Si se detectan landmarks (puntos clave)
    if results.pose_landmarks:
        # Dibuja el esqueleto del cuerpo
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Convierte los landmarks a una lista de coordenadas
        landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

        # Si hay landmarks anteriores, calcula el movimiento
        if prev_landmarks:
            for i, (curr, prev) in enumerate(zip(landmarks, prev_landmarks)):
                # Calcula el desplazamiento
                dx, dy = (curr[0] - prev[0]) * frame.shape[1], (curr[1] - prev[1]) * frame.shape[0]

                # Si el movimiento es significativo, dibuja una línea
                if abs(dx) > 5 or abs(dy) > 5:
                    x, y = int(curr[0] * frame.shape[1]), int(curr[1] * frame.shape[0])
                    cv2.arrowedLine(frame, (int(prev[0] * frame.shape[1]), int(prev[1] * frame.shape[0])),
                                    (x, y), (0, 255, 255), 2, tipLength=0.3)

        # Actualiza las posiciones anteriores de los landmarks
        prev_landmarks = landmarks
    else:
        prev_landmarks = None

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Instant Motion Tracking en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Instant Motion Tracking en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Instant Motion Tracking en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
