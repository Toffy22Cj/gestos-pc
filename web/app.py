from flask import Flask, render_template, Response, jsonify, request
import logging
import sys
from pathlib import Path

# Añadir el root dir al path
sys.path.append(str(Path(__file__).parent.parent))

from core.controller import controller_instance
from db.models import SessionLocal, GestureMapping, add_or_update_mapping, delete_mapping

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({"running": controller_instance.is_running})

@app.route('/api/start', methods=['POST'])
def start_controller():
    controller_instance.start()
    return jsonify({"status": "started"})
    
@app.route('/api/stop', methods=['POST'])
def stop_controller():
    controller_instance.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/mappings', methods=['GET'])
def get_mappings():
    with SessionLocal() as session:
        gestures = session.query(GestureMapping).all()
        return jsonify([{
            "id": g.id,
            "gesture_name": g.gesture_name,
            "command_hyprland": g.command_hyprland or "",
            "command_windows": g.command_windows or "",
            "command_generic": g.command_generic or ""
        } for g in gestures])

@app.route('/api/mappings', methods=['POST'])
def update_mapping():
    data = request.json
    add_or_update_mapping(
        gesture_name=data['gesture_name'],
        hyprland=data.get('command_hyprland'),
        windows=data.get('command_windows'),
        generic=data.get('command_generic')
    )
    return jsonify({"status": "success"})

@app.route('/api/mappings/<gesture_name>', methods=['DELETE'])
def remove_mapping(gesture_name):
    success = delete_mapping(gesture_name)
    if success:
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Gesto no encontrado"}), 404

@app.route('/api/debug_logs', methods=['GET'])
def get_debug_logs():
    return jsonify({
        "last_gesture": controller_instance.last_detected_gesture,
        "logs": list(controller_instance.action_logs)
    })

def gen_frames():
    while True:
        if not controller_instance.is_running:
            import time
            time.sleep(1) # Dormir si no hay cámara
            continue
            
        frame = controller_instance.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            import time
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Empezar automáticamente en background por comodidad del user
    controller_instance.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
