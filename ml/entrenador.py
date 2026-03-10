import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    dataset_path = Path(__file__).parent / "dataset" / "hand_gestures.csv"
    model_path = Path(__file__).parent.parent / "custom_gesture_model.pkl"
    
    if not dataset_path.exists():
        logger.error(f"No se encontró dataset en {dataset_path}. Corre colector.py primero.")
        return
        
    logger.info("Cargando dataset...")
    df = pd.read_csv(dataset_path)
    
    features = df.drop("label", axis=1)
    labels = df["label"]
    
    logger.info(f"Total de muestras: {len(df)}")
    logger.info(f"Gestos en dataset: {labels.unique().tolist()}")
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    logger.info("Entrenando RandomForestClassifier...")
    # RandomForest es rápido, no requiere escalado como SVC y es bastante inmune al ruido
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logger.info(f"💪 Precisión en Validación: {acc * 100:.2f}%")
    
    # Entrenar en todo el dataset final
    rf.fit(features, labels)
    joblib.dump(rf, model_path)
    logger.info(f"✅ Modelo custom guardado en {model_path}")

if __name__ == "__main__":
    train_model()
