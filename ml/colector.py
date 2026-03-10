import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import time
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_data(gesture_name: str, num_samples: int = 500):
    """Recolecta landmarks de la mano y los guarda en un CSV"""
    output_dir = Path(__file__).parent / "dataset"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "hand_gestures.csv"

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1)
        
    try:
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}. Copia hand_landmarker.task a esta carpeta.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("No se pudo iniciar la cámara")
        return

    logger.info(f"== RECOLECCIÓN PARA '{gesture_name}' ==")
    logger.info("Presiona 'r' para empezar a grabar (Tienes 3 segundos para prepararte). Presiona 'q' para salir.")
    
    recording = False
    samples_collected = 0
    data = []

    while cap.isOpened() and samples_collected < num_samples:
        success, image = cap.read()
        if not success: continue
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR@RGB) # Se arreglara a 2RGB mas adelante
        
        # UI
        display_img = image.copy()
        cv2.putText(display_img, f"Gesto: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(display_img, f"Muestras: {samples_collected}/{num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        if recording:
            cv2.putText(display_img, "RECORDING...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = detector.detect_for_video(mp_image, int(time.time() * 1000))
            
            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                # Extraer coordenadas [x0,y0,z0, x1,y1,z1...]
                row = [gesture_name]
                for lm in hand:
                    row.extend([lm.x, lm.y, lm.z])
                data.append(row)
                samples_collected += 1
                
                # Feedback visual
                cv2.circle(display_img, (int(hand[0].x * image.shape[1]), int(hand[0].y * image.shape[0])), 10, (0,255,0), -1)
        else:
             cv2.putText(display_img, "Presiona 'r' para empezar...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Recolector de Gestos", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not recording:
            logger.info("Preparando...")
            cv2.waitKey(3000)
            recording = True
            logger.info("¡Grabando!")

    cap.release()
    cv2.destroyAllWindows()
    
    if len(data) > 0:
        # Generar nombres de columnas
        columns = ["label"]
        for i in range(21):
            columns.extend([f"x{i}", f"y{i}", f"z{i}"])
            
        df_new = pd.DataFrame(data, columns=columns)
        
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
            
        df_final.to_csv(csv_path, index=False)
        logger.info(f"✅ Guardadas {len(data)} muestras en {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gesto", type=str, help="Nombre del gesto a grabar (ej. MI_GESTO_RARO)")
    parser.add_argument("--muestras", type=int, default=500, help="Número de frames a capturar")
    args = parser.parse_args()
    collect_data(args.gesto, args.muestras)
