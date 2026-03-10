import argparse
from db.models import add_or_update_mapping, get_mapping, SessionLocal, GestureMapping
import sys

def list_gestures():
    with SessionLocal() as session:
        gestures = session.query(GestureMapping).all()
        print("\n=== Gestos Configurados ===")
        for g in gestures:
            print(f"- {g.gesture_name}:")
            if g.command_hyprland: print(f"  [Hyprland] {g.command_hyprland}")
            if g.command_windows: print(f"  [Windows] {g.command_windows}")
            if g.command_generic: print(f"  [Generic Linux] {g.command_generic}")
        print("===========================\n")

def main():
    parser = argparse.ArgumentParser(description="CRUD de Gestos para PC")
    parser.add_argument('--list', action='store_true', help='Listar todos los gestos')
    parser.add_argument('--set', type=str, metavar='GESTO', help='Nombre del gesto a configurar (ej. PUNO, PAZ)')
    parser.add_argument('--hyprland', type=str, help='Comando para Hyprland (ej. "exec kitty")')
    parser.add_argument('--windows', type=str, help='Comando para Windows')
    
    args = parser.parse_args()
    
    if args.list:
        list_gestures()
    elif args.set:
        if not args.hyprland and not args.windows:
            print("Error: Debes proveer al menos una acción con --hyprland o --windows")
            sys.exit(1)
        add_or_update_mapping(args.set, hyprland=args.hyprland, windows=args.windows)
        print(f"Gesto '{args.set}' actualizado correctamente.")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
