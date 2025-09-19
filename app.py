from flask import Flask, render_template, request, jsonify, url_for
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import time

app = Flask(__name__)

# ==============================
# Configuración
# ==============================
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

model_path = "model.pkl"
knn = None

# ====== Estadísticas de práctica ======
intentos = 0
correctos = 0
racha = 0

# ==============================
# Rutas
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/entrenamiento")
def entrenamiento():
    return render_template("entrenamiento.html")

@app.route("/practica/<vocal>")
def practica(vocal):
    return render_template("practica.html", vocal=vocal.upper())

# ==============================
# API: guardar landmarks de entrenamiento
# ==============================
@app.route("/guardar_entrenamiento", methods=["POST"])
def guardar_entrenamiento():
    data = request.json
    label = data.get("label")
    landmarks = data.get("landmarks")

    if not label or not landmarks:
        return "Datos incompletos", 400

    folder = os.path.join(dataset_path, label)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{time.time()}.npy")
    np.save(filename, np.array(landmarks))

    return "Guardado ✅"

# ==============================
# API: entrenar modelo
# ==============================
@app.route("/train_model", methods=["POST"])
def train_model():
    global knn
    X, y = [], []
    for label in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    landmarks = np.load(os.path.join(folder, file))
                    X.append(landmarks)
                    y.append(label)

    if len(X) > 0 and len(set(y)) > 1:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        joblib.dump(knn, model_path)
        return "✅ Modelo entrenado"
    else:
        return "⚠️ Datos insuficientes"

# ==============================
# API: reconocer
# ==============================
@app.route("/reconocer", methods=["POST"])
def reconocer():
    global knn
    data = request.json
    landmarks = np.array(data.get("landmarks")).reshape(1, -1)

    if knn is None and os.path.exists(model_path):
        knn = joblib.load(model_path)

    if knn:
        pred = knn.predict(landmarks)[0]
        return jsonify({"prediccion": pred})
    else:
        return jsonify({"prediccion": None})

# ==============================
# API: práctica (estadísticas)
# ==============================
@app.route("/status_practica")
def status_practica():
    global intentos, correctos, racha
    return jsonify({
        "intentos": intentos,
        "correctos": correctos,
        "racha": racha,
        "precision": (correctos / intentos * 100) if intentos > 0 else 0
    })

@app.route("/reiniciar_practica", methods=["POST"])
def reiniciar_practica():
    global intentos, correctos, racha
    intentos, correctos, racha = 0, 0, 0
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
