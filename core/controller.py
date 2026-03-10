import cv2
import time
import logging
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.executors import get_executor, WindowsExecutor, HyprlandExecutor
from gestos.extractor import HandFeatureExtractor, FaceFeatureExtractor
from db.models import get_mapping

logger = logging.getLogger(__name__)

class GestureController:
    """Controlador singleton-like que maneja la cámara y los modelos de AI en un hilo separado"""
    def __init__(self):
        self.is_running = False
        self.cap = None
        self.current_frame = None
        
        # Variables de Debug/Logs para la Web UI
        self.last_detected_gesture = "Ninguno"
        self.action_logs = deque(maxlen=20) # Guardar últimos 20 eventos
        
        # Ejecutores y Extractores
        self.executor = get_executor()
        self.hand_extractor = HandFeatureExtractor(history_length=12)
        self.face_extractor = FaceFeatureExtractor(history_length=12)
        
        # Opciones ML
        self.hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1)
            
        self.face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1)

    def start(self):
        if self.is_running:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("No se pudo iniciar la cámara en el Controller")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Controlador de Gestos Iniciado (Background)")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.current_frame = None
        logger.info("Controlador de Gestos Detenido")

    def get_frame(self):
        """Devuelve el frame actual codificado en JPG para la Web UI"""
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if ret:
                return buffer.tobytes()
        return None

    def _draw_landmarks(self, image, hand_landmarks):
        # Implementación simplificada solo para UI web
        h, w, c = image.shape
        for i, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    def _draw_face_landmarks(self, image, face_landmarks):
        h, w, c = image.shape
        # Dibujar solo algunos puntos clave de la cara para no saturar la imagen
        for i, landmark in enumerate(face_landmarks):
            if i % 5 == 0: # Dibujar 1 de cada 5 puntos
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 1, (0, 255, 255), cv2.FILLED)

    def _run_loop(self):
        try:
            hand_detector = vision.HandLandmarker.create_from_options(self.hand_options)
            face_detector = vision.FaceLandmarker.create_from_options(self.face_options)
        except Exception as e:
            logger.error(f"Error AI: {e}")
            return

        last_gesture = None
        last_action_time = 0
        COOLDOWN = 2.0

        while self.is_running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
                
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Inferencias
            hand_results = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
            face_results = face_detector.detect_for_video(mp_image, frame_timestamp_ms)
            
            current_gesture = None
            
            if hand_results.hand_landmarks:
                first_hand = hand_results.hand_landmarks[0]
                self._draw_landmarks(image, first_hand)
                current_gesture = self.hand_extractor.get_gesture(first_hand)
                
            if not current_gesture and face_results.face_landmarks:
                first_face = face_results.face_landmarks[0]
                self._draw_face_landmarks(image, first_face)
                current_gesture = self.face_extractor.get_gesture(first_face)
                
            if current_gesture:
                self.last_detected_gesture = current_gesture
                cv2.putText(image, current_gesture, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                self.last_detected_gesture = "Ninguno"
                            
            # Ejecutor
            current_time = time.time()
            if current_gesture and current_gesture != "DESCONOCIDO":
                is_dynamic = "SWIPE" in current_gesture
                can_trigger = False
                
                if is_dynamic:
                    # Los gestos dinámicos operan por Cooldown (para evitar disparar 10 swipes en 1 segundo de fleteo)
                    if (current_time - last_action_time) > COOLDOWN:
                        can_trigger = True
                else:
                    # Los gestos estáticos operan por LATCH (candado). Se ejecutan UNA vez, 
                    # y no se repiten hasta que el gesto se pierda o cambie.
                    if current_gesture != last_gesture:
                        can_trigger = True
                
                if can_trigger:
                    mapping = get_mapping(current_gesture)
                    if mapping:
                        cmd = None
                        if isinstance(self.executor, HyprlandExecutor): cmd = mapping.command_hyprland
                        elif isinstance(self.executor, WindowsExecutor): cmd = mapping.command_windows
                        else: cmd = mapping.command_generic
                        
                        if cmd:
                            timestamp = time.strftime("%H:%M:%S")
                            log_msg = f"[{timestamp}] {current_gesture} -> {cmd}"
                            self.action_logs.append(log_msg)
                            logger.info(f"UI Executing ({'Dynamic' if is_dynamic else 'Static'}): {cmd}")
                            self.executor.execute(cmd)
                            
                    last_action_time = current_time
                
                last_gesture = current_gesture
            elif not current_gesture:
                last_gesture = None
                
            self.current_frame = image
            time.sleep(0.01) # Liberar CPU
            
        hand_detector.close()
        face_detector.close()

# Instancia global
controller_instance = GestureController()
