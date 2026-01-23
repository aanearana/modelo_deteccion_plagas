from flask import Flask, request, jsonify, send_file
from pathlib import Path
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

_model = None

def get_model():
    global _model
    if _model is None:
        print("Cargando YOLO...")
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
    return _model

@app.route("/")
def home():
    return jsonify({"status": "ok"})

@app.route("/health")  
def health():
    return "OK", 200

@app.route("/detect", methods=["POST"])
def detect_plagas():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Invalid file"}), 400
        
        print("Obteniendo modelo...")
        model = get_model()
        print("Modelo obtenido")
        
        img_path = UPLOAD_FOLDER / file.filename
        print(f"Guardando: {img_path}")
        file.save(img_path)
        
        print("Prediciendo...")
        results = model.predict(source=str(img_path), save=True, conf=0.25)
        print("Predicción hecha")
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({
                "plaga": cls_name, 
                "confianza": round(conf, 2)
            })
        
        return jsonify({
            "detecciones": detections,
            "total": len(detections),
            "imagen_resultado": f"/imagen/{file.filename}"
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/imagen/<filename>")
def get_imagen(filename):
    try:
        img_path = Path("runs/detect/predict") / filename
        if not img_path.exists():
            return jsonify({"error": "Not found"}), 404
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Iniciando servidor en puerto {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)