from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from pathlib import Path
import os
import sys

app = Flask(__name__)

# Obtener directorio de trabajo actual
WORK_DIR = Path(os.getcwd())

# Configurar modelo - descargar si no existe o está corrupto
MODEL_PATH = Path("best.pt")

try:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 10_000_000:  # Si < 10 MB, descarga
        print(f"[INFO] Modelo no encontrado o corrupto. Descargando YOLOv8n...", file=sys.stderr)
        model = YOLO('yolov8n.pt')  # Descargar modelo nano preentrenado
        model.save(str(MODEL_PATH))
        print(f"[INFO] Modelo guardado en {MODEL_PATH}", file=sys.stderr)
    else:
        print(f"[INFO] Cargando modelo desde {MODEL_PATH}", file=sys.stderr)
        model = YOLO(str(MODEL_PATH))
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}", file=sys.stderr)
    print(f"[INFO] Descargando YOLOv8n como respaldo...", file=sys.stderr)
    model = YOLO('yolov8n.pt')

# Carpeta para subir imágenes
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API de detección de plagas activa. Usa /predict_image o /detect."})

@app.route("/predict_image")
def predict_image():
    """Predice plagas en una imagen que ya está en la carpeta del proyecto"""
    filename = request.args.get('file')
    
    if not filename:
        return jsonify({"error": "Parámetro 'file' no especificado. Uso: /predict_image?file=nombre.jpg"}), 400
    
    # Buscar el archivo en el directorio de trabajo
    img_path = WORK_DIR / filename
    
    if not img_path.exists():
        archivos = list(WORK_DIR.glob("*.jpg")) + list(WORK_DIR.glob("*.png"))
        archivos_nombres = [f.name for f in archivos]
        return jsonify({
            "error": f"Archivo no encontrado: {filename}",
            "directorio": str(WORK_DIR),
            "archivos_disponibles": archivos_nombres
        }), 404
    
    try:
        # Ejecutar predicción
        results = model.predict(source=str(img_path), save=True)
        
        # Imagen con detecciones
        output_path = Path(results[0].save_dir) / filename
        
        # Extraer detecciones
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"plaga": cls_name, "confianza": round(conf, 2)})
        
        return jsonify({
            "detecciones": detections,
            "imagen_resultado": str(output_path),
            "archivo_procesado": filename
        })
    except Exception as e:
        return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 400
@app.route("/detect", methods=["POST"])
def detect_plagas():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa multipart/form-data con campo 'file'"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400

    # Guardar la imagen subida
    img_path = UPLOAD_FOLDER / file.filename
    file.save(img_path)

    # Ejecutar predicción
    results = model.predict(source=str(img_path), save=True)

    # Imagen con detecciones
    output_path = Path(results[0].save_dir) / file.filename

    # Extraer detecciones
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"plaga": cls_name, "confianza": round(conf, 2)})

    return jsonify({
        "detecciones": detections,
        "imagen_resultado": str(output_path)
    })

@app.route("/image/<filename>")
def get_image(filename):
    path = Path("runs/detect/predict") / filename
    if not path.exists():
        return jsonify({"error": "Imagen no encontrada"}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint para subir imagen - solo POST con multipart/form-data"""
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa multipart/form-data con campo 'file'"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400

    # Guardar la imagen subida
    img_path = UPLOAD_FOLDER / file.filename
    file.save(img_path)

    # Ejecutar predicción
    results = model.predict(source=str(img_path), save=True)

    # Imagen con detecciones
    output_path = Path(results[0].save_dir) / file.filename

    # Extraer detecciones
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"plaga": cls_name, "confianza": round(conf, 2)})

    return jsonify({
        "detecciones": detections,
        "imagen_resultado": str(output_path)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)