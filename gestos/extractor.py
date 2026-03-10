import math
import os
import joblib
import warnings
from typing import List, Optional, Tuple
from collections import deque
from pathlib import Path

# Suprimir avisos de scikit-learn si no hay feature names
warnings.filterwarnings("ignore", category=UserWarning)

class HandFeatureExtractor:
    """Extrae características de alto nivel (gestos) a partir de los landmarks de MediaPipe."""
    
    def __init__(self, history_length=12):
        # Índices de los landmarks en MediaPipe
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        self.INDEX_FINGER_MCP = 5
        self.MIDDLE_FINGER_MCP = 9
        self.RING_FINGER_MCP = 13
        self.PINKY_MCP = 17
        
        # Historial para suavizado (debouncing)
        self.history = deque(maxlen=history_length)
        
        # Carga del modelo ML Customizado (si existe)
        self.ml_model = None
        model_path = Path(__file__).parent.parent / "custom_gesture_model.pkl"
        if model_path.exists():
             try:
                 self.ml_model = joblib.load(model_path)
             except Exception:
                 pass

    def _get_distance(self, pt1, pt2) -> float:
        """Calcula la distancia euclidiana entre dos landmarks (x, y)."""
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

    def _is_finger_closed(self, tip, mcp, wrist) -> bool:
        """Determina si un dedo (excepto el pulgar) está cerrado comparando la distancia
        de la punta a la muñeca, contra la distancia del nudillo a la muñeca."""
        dist_tip_to_wrist = self._get_distance(tip, wrist)
        dist_mcp_to_wrist = self._get_distance(mcp, wrist)
        # Si la punta está más cerca de la muñeca que el nudillo inferior, el dedo está doblado
        return dist_tip_to_wrist < dist_mcp_to_wrist

    def get_gesture(self, hand_landmarks) -> Optional[str]:
        """Devuelve un string con el nombre del gesto reconocido."""
        if not hand_landmarks:
            return None
            
        # 1. Pipeline de Machine Learning (Prioridad si el usuario entrenó su modelo)
        if self.ml_model:
            row = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
            prediction = self.ml_model.predict([row])[0]
            return self._smooth_gesture(prediction)
            
        # 2. Pipeline Heurístico (Si no hay modelo, recae en matemáticas base)
        wrist = hand_landmarks[self.WRIST]
        
        # Gestos basados en distancias de puntas
        thumb_tip = hand_landmarks[self.THUMB_TIP]
        middle_tip = hand_landmarks[self.MIDDLE_FINGER_TIP]
        
        # Distancia entre la punta del pulgar y el dedo medio
        # Aumentamos el umbral para que detecte el cierre más fácilmente
        dist_thumb_middle = self._get_distance(thumb_tip, middle_tip)
        
        if dist_thumb_middle < 0.09:
            return self._smooth_gesture("MEDIO_Y_PULGAR_JUNTOS")
        
        # Evaluar estado de cada dedo (abierto/cerrado)
        index_closed = self._is_finger_closed(
            hand_landmarks[self.INDEX_FINGER_TIP], 
            hand_landmarks[self.INDEX_FINGER_MCP], 
            wrist)
            
        middle_closed = self._is_finger_closed(
            hand_landmarks[self.MIDDLE_FINGER_TIP], 
            hand_landmarks[self.MIDDLE_FINGER_MCP], 
            wrist)
            
        ring_closed = self._is_finger_closed(
            hand_landmarks[self.RING_FINGER_TIP], 
            hand_landmarks[self.RING_FINGER_MCP], 
            wrist)
            
        pinky_closed = self._is_finger_closed(
            hand_landmarks[self.PINKY_TIP], 
            hand_landmarks[self.PINKY_MCP], 
            wrist)
            
        # Lógica heurística de gestos básicos
        fingers_closed = [index_closed, middle_closed, ring_closed, pinky_closed]
        closed_count = sum(fingers_closed)
        
        raw_gesture = "DESCONOCIDO"
        if closed_count == 4:
            raw_gesture = "PUNO"
        elif closed_count == 0:
            raw_gesture = "PALMA_ABIERTA"
        elif not index_closed and middle_closed and ring_closed and pinky_closed:
            raw_gesture = "APUNTAR"
        elif not index_closed and not middle_closed and ring_closed and pinky_closed:
            raw_gesture = "PAZ"
            
        return self._smooth_gesture(raw_gesture)

    def _smooth_gesture(self, new_gesture: str) -> Optional[str]:
        """Añade el nuevo gesto al buffer e intenta devolver el más repetido 
        para evitar falsos positivos o saltos repentinos (flickering)."""
        self.history.append(new_gesture)
        
        if len(self.history) < self.history.maxlen:
            return None # Aún no tenemos suficientes frames para estar seguros
            
        # Contamos cuál es el gesto mayoritario en los últimos X frames
        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
            
        most_common = max(counts.items(), key=lambda x: x[1])
        
        # Si el gesto más común ocupa más del 70% de los frames (ej. 9 de 12), decídelo.
        if most_common[1] >= (self.history.maxlen * 0.7):
            return most_common[0]
        else:
            return None # Indeciso

class FaceFeatureExtractor:
    """Extrae gestos faciales a partir de la malla FaceLandmarks de MediaPipe"""
    
    def __init__(self, history_length=12):
        self.history = deque(maxlen=history_length)
        
        # Puntos clave Face Mesh (Aproximaciones estándar de MediaPipe)
        self.LEFT_EYE_TOP = 159      # Parte superior del ojo izquierdo
        self.LEFT_EYE_BOTTOM = 145   # Parte inferior
        self.LEFT_EYEBROW_TOP = 65   # Punto alto de la ceja izquierda
        self.NOSE_TIP = 1            # Punta de la nariz (como referencia base)
        
    def _get_distance(self, pt1, pt2) -> float:
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)
        
    def get_gesture(self, face_landmarks) -> Optional[str]:
        if not face_landmarks:
            return None
            
        left_eye_top = face_landmarks[self.LEFT_EYE_TOP]
        left_eyebrow_top = face_landmarks[self.LEFT_EYEBROW_TOP]
        left_eye_bottom = face_landmarks[self.LEFT_EYE_BOTTOM]
        
        # Distancia entre la ceja y el ojo (normalizada por el tamaño del ojo para compensar distanciamiento de la cámara)
        eye_height = self._get_distance(left_eye_top, left_eye_bottom)
        eyebrow_distance = self._get_distance(left_eyebrow_top, left_eye_top)
        
        # Si la distancia de la ceja al ojo es más de ~2.2 veces el tamaño del ojo, está muy levantada
        # (Este umbral puede requerir ajustes según la persona)
        ratio = eyebrow_distance / (eye_height + 0.0001) # +0.0001 para evitar div por cero
        
        raw_gesture = "DESCONOCIDO"
        if ratio > 2.4:
            raw_gesture = "CEJA_IZQ_ARRIBA"
            
        return self._smooth_gesture(raw_gesture)

    def _smooth_gesture(self, new_gesture: str) -> Optional[str]:
        self.history.append(new_gesture)
        if len(self.history) < self.history.maxlen:
            return None
            
        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
            
        most_common = max(counts.items(), key=lambda x: x[1])
        if most_common[1] >= (self.history.maxlen * 0.7):
            return most_common[0]
        return None
