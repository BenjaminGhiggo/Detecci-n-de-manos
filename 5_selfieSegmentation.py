import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Color de fondo (modifícalo para cambiar el color de fondo)
BACKGROUND_COLOR = (0, 255, 0)  # Verde

# Imagen de fondo personalizada (comenta esta línea si no quieres usar una imagen)
background_image = cv2.imread('fondo.jpg')

# Captura de video desde la webcam (0 para la webcam predeterminada)
cap = cv2.VideoCapture(0)

# Habilita la ventana en modo redimensionable
cv2.namedWindow("Segmentación de Fondo en Tiempo Real", cv2.WINDOW_NORMAL)

# Ajusta la ventana al tamaño de la pantalla (sin ocultar botones)
cv2.moveWindow("Segmentación de Fondo en Tiempo Real", 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convierte la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)

    # Extrae la máscara de segmentación
    mask = results.segmentation_mask
    condition = mask > 0.5

    # Si hay una imagen de fondo, redimensionarla al tamaño del frame
    if background_image is not None:
        background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    else:
        # Si no hay imagen, usar un color sólido como fondo
        background = np.zeros(frame.shape, dtype=np.uint8)
        background[:] = BACKGROUND_COLOR

    # Combina el fondo y el primer plano usando la máscara de segmentación
    output_image = np.where(condition[..., None], frame, background)

    # Obtiene el tamaño actual de la ventana
    window_width = cv2.getWindowImageRect("Segmentación de Fondo en Tiempo Real")[2]
    window_height = cv2.getWindowImageRect("Segmentación de Fondo en Tiempo Real")[3]

    # Redimensiona el frame al tamaño de la ventana
    resized_frame = cv2.resize(output_image, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Muestra el video en tiempo real
    cv2.imshow("Segmentación de Fondo en Tiempo Real", resized_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
