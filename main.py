import cv2
import mediapipe as mp

# Rutas de video
input_video_path = "video/video1.mp4"
output_video_path = "video_procesado/video1_procesado.mp4"

# Inicializa MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configuración del video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Si se detectan manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Escribe el frame en el video de salida
    out.write(frame)

    # (Opcional) Muestra el video en tiempo real
    cv2.imshow("Video Procesado", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
out.release()
cv2.destroyAllWindows()
