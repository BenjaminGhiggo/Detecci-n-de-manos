import cv2
import mediapipe as mp

# Inicializa MediaPipe Face Stylization y Drawing
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Colores para estilización (modifica estos colores para personalizar el estilo)
FACEMESH_COLOR = (0, 255, 0)    # Verde para la malla
CONTOUR_COLOR = (0, 0, 255)     # Rojo para el contorno

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Estilización Facial en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Estilización Facial en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Si se detectan rostros
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibuja la malla facial con un color personalizado
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=FACEMESH_COLOR, thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=FACEMESH_COLOR, thickness=1, circle_radius=1)
            )

            # Dibuja el contorno del rostro con un color diferente
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=CONTOUR_COLOR, thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=CONTOUR_COLOR, thickness=2, circle_radius=1)
            )

    # Aplica un filtro para estilización (por ejemplo, efecto de dibujo a lápiz)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    frame = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Estilización Facial en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Estilización Facial en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Estilización Facial en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
