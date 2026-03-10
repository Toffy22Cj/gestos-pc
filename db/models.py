from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Definir la ruta de la base de datos dentro del proyecto
DB_PATH = Path(__file__).parent.parent / 'gestos_db.sqlite3'
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

Base = declarative_base()

class GestureMapping(Base):
    """Modelo para mapear un gesto a un comando por SO"""
    __tablename__ = 'gesture_mappings'

    id = Column(Integer, primary_key=True)
    gesture_name = Column(String(50), nullable=False, unique=True)
    command_hyprland = Column(String(200), nullable=True)
    command_windows = Column(String(200), nullable=True)
    command_generic = Column(String(200), nullable=True)

    def __repr__(self):
        return f"<GestureMapping(gesture='{self.gesture_name}', hypr='{self.command_hyprland}')>"

# Crear las tablas
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

def get_mapping(gesture_name: str) -> GestureMapping:
    """Busca el mapeo de un gesto específico"""
    with SessionLocal() as session:
        return session.query(GestureMapping).filter_by(gesture_name=gesture_name).first()

def add_or_update_mapping(gesture_name: str, hyprland=None, windows=None, generic=None):
    """Añade o actualiza un mapeo de gesto"""
    with SessionLocal() as session:
        mapping = session.query(GestureMapping).filter_by(gesture_name=gesture_name).first()
        if not mapping:
            mapping = GestureMapping(gesture_name=gesture_name)
            session.add(mapping)
            
        if hyprland is not None:
            mapping.command_hyprland = hyprland
        if windows is not None:
            mapping.command_windows = windows
        if generic is not None:
            mapping.command_generic = generic
            
        session.commit()
        logger.info(f"Mapeo guardado: {gesture_name}")

def delete_mapping(gesture_name: str) -> bool:
    """Elimina un mapeo de gesto. Devuelve True si fue eliminado."""
    with SessionLocal() as session:
        mapping = session.query(GestureMapping).filter_by(gesture_name=gesture_name).first()
        if mapping:
            session.delete(mapping)
            session.commit()
            logger.info(f"Mapeo eliminado: {gesture_name}")
            return True
        return False

def init_default_mappings():
    """Inicializa la base de datos con los mapeos por defecto si está vacía."""
    with SessionLocal() as session:
        count = session.query(GestureMapping).count()
        if count == 0:
            logger.info("Inicializando BD con gestos por defecto...")
            # PUNO -> Siguiente workspace
            add_or_update_mapping("PUNO", hyprland="exec hyprctl dispatch workspace e+1")
            
            # PALMA ABIERTA -> Abrir terminal (Kitty)
            add_or_update_mapping("PALMA_ABIERTA", hyprland="exec kitty")
            
            # PAZ -> Anterior workspace
            add_or_update_mapping("PAZ", hyprland="exec hyprctl dispatch workspace e-1")
            
            # APUNTAR -> Cerrar ventana activa
            add_or_update_mapping("APUNTAR", hyprland="exec hyprctl dispatch killactive")
            
if __name__ == "__main__":
    init_default_mappings()
    print("Base de datos inicializada correctaente en db/gestos_db.sqlite3")
