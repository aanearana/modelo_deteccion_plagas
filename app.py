from flask import Flask, request, jsonify
from pathlib import Path
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Lazy loading - modelo global
_model = None

def get_model():
    global _model
    if _model is None:
        print("Cargando modelo YOLO...")
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
    return _model

@app.route("/")
def home():
    return jsonify({"status": "ok"})

@app.route("/detect", methods=["POST"])
def detect_plagas():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Invalid file"}), 400
        
        model = get_model()
        
        img_path = UPLOAD_FOLDER / file.filename
        file.save(img_path)
        
        results = model.predict(source=str(img_path), save=True, conf=0.25)
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"plaga": cls_name, "confianza": round(conf, 2)})
        
        # Ruta de la imagen con detecciones dibujados
        output_dir = Path(results[0].save_dir)
        output_img = output_dir / file.filename
        
        return jsonify({
            "detecciones": detections,
            "total": len(detections),
            "imagen_resultado": f"/imagen/{file.filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/imagen/<filename>")
def get_imagen(filename):
    """Devuelve la imagen con detecciones dibujados"""
    try:
        img_path = Path("runs/detect/predict") / filename
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404
        from flask import send_file
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)