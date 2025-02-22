import cv2
import mediapipe as mp

# Inicializa MediaPipe Objectron y Drawing
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name='Cup'  # Modelos disponibles: 'Cup', 'Chair', 'Shoe', 'Camera'
)

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Detección de Objetos en 3D en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Detección de Objetos en 3D en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = objectron.process(image_rgb)

    # Si se detectan objetos
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Dibuja el cuadro delimitador en 3D y puntos clave
            mp_drawing.draw_landmarks(
                frame, 
                detected_object.landmarks_2d, 
                mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Detección de Objetos en 3D en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Detección de Objetos en 3D en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Detección de Objetos en 3D en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
