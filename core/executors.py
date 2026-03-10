import platform
import subprocess
import logging

logger = logging.getLogger(__name__)

class ActionExecutor:
    """Clase base abstracta para ejecutar acciones en el sistema"""
    def execute(self, action_command: str):
        raise NotImplementedError("Debe implementarse en la subclase")

class WindowsExecutor(ActionExecutor):
    def execute(self, action_command: str):
        logger.info(f"[Windows] Ejecutando: {action_command}")
        # Aquí iría la lógica usando pyautogui o os.system

class HyprlandExecutor(ActionExecutor):
    def execute(self, action_command: str):
        logger.info(f"[Hyprland] Dispatching: {action_command}")
        try:
            # action_command será algo como "exec kitty" o "workspace e+1"
            # Separamos en lista para subprocess
            cmd_args = ["hyprctl", "dispatch"] + action_command.split(" ", 1)
            subprocess.run(cmd_args, check=True)
        except Exception as e:
            logger.error(f"Error ejecutando hyprctl: {e}")

class GenericLinuxExecutor(ActionExecutor):
    def execute(self, action_command: str):
        logger.info(f"[Linux Genérico] Ejecutando: {action_command}")
        # Aquí iría lógica con xdotool u otros

def get_executor() -> ActionExecutor:
    """Devuelve el ejecutor adecuado según el SO actual."""
    os_name = platform.system()
    if os_name == "Windows":
        return WindowsExecutor()
    elif os_name == "Linux":
        # Forma rudimentaria de detectar Hyprland
        import os
        if "HYPRLAND_INSTANCE_SIGNATURE" in os.environ:
            return HyprlandExecutor()
        else:
            return GenericLinuxExecutor()
    else:
        logger.warning(f"OS no soportado: {os_name}. Usando logger genérico.")
        return ActionExecutor()
