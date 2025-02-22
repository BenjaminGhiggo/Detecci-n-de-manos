import cv2
import mediapipe as mp

# Inicializa MediaPipe Pose y Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Detección de Postura Corporal en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Detección de Postura Corporal en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Si se detectan puntos de referencia del cuerpo
    if results.pose_landmarks:
        # Dibuja los puntos y conexiones del cuerpo
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Detección de Postura Corporal en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Detección de Postura Corporal en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Detección de Postura Corporal en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
