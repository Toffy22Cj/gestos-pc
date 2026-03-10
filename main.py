import cv2
import time
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.executors import get_executor, WindowsExecutor, HyprlandExecutor
from gestos.extractor import HandFeatureExtractor, FaceFeatureExtractor
from db.models import get_mapping

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conexiones de las manos (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8), # Índice
    (5, 9), (9, 10), (10, 11), (11, 12), # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Menique
]

def draw_landmarks(image, hand_landmarks):
    """Dibuja los puntos y conexiones de la mano (Alternativa a mp_drawing)"""
    h, w, c = image.shape
    # Dibujar conexiones
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_pt = hand_landmarks[start_idx]
        end_pt = hand_landmarks[end_idx]
        
        cv2.line(image, 
                 (int(start_pt.x * w), int(start_pt.y * h)), 
                 (int(end_pt.x * w), int(end_pt.y * h)), 
                 (0, 255, 0), 2)
                 
    # Dibujar puntos
    for i, landmark in enumerate(hand_landmarks):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

def main():
    logger.info("Iniciando Gestos PC con Tasks API...")
    
    executor = get_executor()
    logger.info(f"Ejecutor seleccionado: {executor.__class__.__name__}")
    
    # -----------------------------
    # 1. Configurar Hand Landmarker
    # -----------------------------
    hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1) # Bajamos a 1 mano por rendimiento
        
    # -----------------------------
    # 2. Configurar Face Landmarker
    # -----------------------------
    face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1)
    
    try:
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        face_detector = vision.FaceLandmarker.create_from_options(face_options)
    except Exception as e:
        logger.error(f"Error cargando los modelos: {e}. Asegúrate de tener hand_landmarker.task y face_landmarker.task")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara web (índice 0).")
        return

    logger.info("Presiona 'q' para salir.")
    
    # Inicializar los extractores heurísticos
    hand_extractor = HandFeatureExtractor(history_length=12)
    face_extractor = FaceFeatureExtractor(history_length=12)
    
    # Variables para evitar spam de comandos (Debouncing local de ejecutores)
    last_gesture = None
    last_action_time = 0
    COOLDOWN = 2.0 # Subimos a 2 segundos de gracia entre enviar el mismo comando

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Espejo e RGB
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a formato MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detectar Manos y Caras
            frame_timestamp_ms = int(time.time() * 1000)
            hand_results = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
            face_results = face_detector.detect_for_video(mp_image, frame_timestamp_ms)
            
            current_gesture = None
            
            # --- 1. Analizar Manos ---
            if hand_results.hand_landmarks:
                first_hand = hand_results.hand_landmarks[0]
                draw_landmarks(image, first_hand)
                current_gesture = hand_extractor.get_gesture(first_hand)
                
            # --- 2. Analizar Caras (Si no hubo gesto de mano que procesar) ---
            if not current_gesture and face_results.face_landmarks:
                first_face = face_results.face_landmarks[0]
                current_gesture = face_extractor.get_gesture(first_face)
                
            # --- Dibujar Gesto ---
            if current_gesture:
                cv2.putText(image, current_gesture, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
            # Ejecutar acciones leyendo desde la Base de Datos
            current_time = time.time()
            if current_gesture and current_gesture != "DESCONOCIDO":
                if current_gesture != last_gesture or (current_time - last_action_time) > COOLDOWN:
                    logger.info(f"Gesto Detectado: {current_gesture}")
                    
                    # Buscar en BD
                    mapping = get_mapping(current_gesture)
                    if mapping:
                        command_to_run = None
                        if isinstance(executor, HyprlandExecutor):
                            command_to_run = mapping.command_hyprland
                        elif isinstance(executor, WindowsExecutor):
                            command_to_run = mapping.command_windows
                        else:
                            command_to_run = mapping.command_generic
                            
                        if command_to_run:
                            executor.execute(command_to_run)
                        else:
                            logger.info(f"No hay comando configurado para {current_gesture} en este SO.")
                    else:
                        logger.warning(f"Gesto {current_gesture} no encontrado en Base de Datos.")
                        
                    last_gesture = current_gesture
                    last_action_time = current_time
            elif not current_gesture:
                last_gesture = None
                    
            cv2.imshow('Gestos PC - Deteccion', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario.")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Recursos liberados, cerrando.")

if __name__ == "__main__":
    main()
