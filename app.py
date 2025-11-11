import os, json, io
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import make_gradcam_heatmap, overlay_heatmap
from utils import load_and_preprocess

disease_suggestions = {
    "bacterial_blight": {
        "name": "Bacterial Leaf Blight",
        "cause": "Caused by Xanthomonas oryzae pv. oryzae bacteria, leading to wilting and drying of leaves.",
        "treatment": [
            "Use resistant rice varieties (e.g., IR20, IR64).",
            "Avoid excessive nitrogen fertilizer.",
            "Spray copper-based fungicides or bactericides like Streptocycline (0.01%)."
        ],
        "prevention": [
            "Ensure proper field drainage.",
            "Remove infected plants early.",
            "Rotate crops to prevent bacterial buildup."
        ]
    },
    "brown_spot": {
        "name": "Brown Spot Disease",
        "cause": "Caused by the fungus Bipolaris oryzae, forming brown lesions on leaves and panicles.",
        "treatment": [
            "Apply fungicides such as Mancozeb (2.5g/L) or Carbendazim (1g/L).",
            "Improve soil fertility with balanced nutrients (especially nitrogen)."
        ],
        "prevention": [
            "Use disease-free seeds.",
            "Avoid dense planting and waterlogging.",
            "Maintain field sanitation and remove debris."
        ]
    },
    "rice_blast": {
        "name": "Rice Blast Disease",
        "cause": "Caused by Magnaporthe oryzae fungus, leading to spindle-shaped lesions and poor yield.",
        "treatment": [
            "Spray Tricyclazole (0.6g/L) or Isoprothiolane (1.5mL/L).",
            "Apply potassium and silicon fertilizers for resistance."
        ],
        "prevention": [
            "Use resistant varieties (e.g., HR12, Co 39).",
            "Avoid excessive nitrogen during early growth.",
            "Maintain good air circulation in the field."
        ]
    },
    "sheath_blight": {
        "name": "Sheath Blight",
        "cause": "Caused by Rhizoctonia solani fungus, forming irregular lesions near waterline.",
        "treatment": [
            "Spray Hexaconazole (1mL/L) or Validamycin (2mL/L).",
            "Improve field drainage and reduce humidity."
        ],
        "prevention": [
            "Use moderate plant spacing.",
            "Avoid continuous rice cropping.",
            "Burn or remove infected residues."
        ]
    },
    "leaf_smut": {
        "name": "Leaf Smut",
        "cause": "Caused by Entyloma oryzae fungus, producing black spots and streaks on leaves.",
        "treatment": [
            "Spray Propiconazole (1mL/L) or Mancozeb (2.5g/L).",
            "Apply balanced NPK fertilizers."
        ],
        "prevention": [
            "Use resistant rice varieties.",
            "Maintain proper spacing and weed control."
        ]
    },
    "healthy": {
        "name": "Healthy Leaf",
        "cause": "No signs of infection detected.",
        "treatment": [
            "Maintain regular irrigation and nutrient balance."
        ],
        "prevention": [
            "Continue good agricultural practices.",
            "Monitor regularly for early disease symptoms."
        ]
    }
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODELS_DIR = 'models'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_indices(path=os.path.join(MODELS_DIR,'class_indices.json')):
    with open(path,'r') as f:
        d = json.load(f)
    return {int(v):k for k,v in d.items()}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    f = request.files['file']
    if f.filename == '':
        return 'No selected file', 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    # Load model
    model = load_model(os.path.join(MODELS_DIR, 'rice_model.h5'))
    img = load_and_preprocess(save_path, target_size=(224,224))
    x = np.expand_dims(img, 0)
    preds = model.predict(x)[0]

    with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'r') as fh:
        mapping = json.load(fh)
    invmap = {int(v): k for k, v in mapping.items()}

    top_idx = int(preds.argmax())
    label = invmap[top_idx]
    score = float(preds[top_idx])

    # Get suggestion
    suggestion = disease_suggestions.get(label, {
        "name": label.title(),
        "cause": "No details available.",
        "treatment": [],
        "prevention": []
    })

    # Build HTML response
    result_html = f"""
    <div style="padding: 15px; font-family: Arial; color: #2D5016;">
        <h2 style="color:#2D5016;">Prediction Result</h2>
        <p><b>Disease:</b> {suggestion['name']}</p>
        <p><b>Confidence:</b> {score:.2%}</p>
        <p><b>Cause:</b> {suggestion['cause']}</p>
        <h3 style="margin-top:10px;">Recommended Treatment:</h3>
        <ul>{"".join(f"<li>{t}</li>" for t in suggestion['treatment'])}</ul>
        <h3>Preventive Measures:</h3>
        <ul>{"".join(f"<li>{p}</li>" for p in suggestion['prevention'])}</ul>
    </div>
    """

    return result_html


    # Grad-CAM (using last conv layer name typical in MobileNetV2)
    last_conv = 'Conv_1' if 'Conv_1' in [l.name for l in model.layers] else model.layers[-3].name
    heatmap = make_gradcam_heatmap(np.expand_dims(img,0), model, last_conv, pred_index=top_idx)
    import cv2
    orig = cv2.cvtColor(cv2.imread(save_path), cv2.COLOR_BGR2RGB)
    overlay = overlay_heatmap(heatmap, orig)
    out_path = os.path.join('static', 'gradcam_' + filename)
    os.makedirs('static', exist_ok=True)
    import matplotlib.pyplot as plt
    plt.imsave(out_path, overlay)

    return render_template('result.html', label=label, score=score, image_url='/' + out_path)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
