import cv2
import mediapipe as mp

# Inicializa MediaPipe Face Detection y Drawing
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Detección de Rostros en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Detección de Rostros en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    # Si se detectan rostros
    if results.detections:
        for detection in results.detections:
            # Dibuja los cuadros de detección en los rostros
            mp_drawing.draw_detection(frame, detection)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Detección de Rostros en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Detección de Rostros en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Detección de Rostros en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
