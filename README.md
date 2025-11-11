# Rice Leaf Disease Detection (Local project)

This project scaffold lets you add datasets, train a CNN using transfer learning (MobileNetV2),
run inference locally, and host a minimal Flask web interface for image upload + Grad-CAM.

## Requirements
- Python 3.8+
- See `requirements.txt` for Python packages
- GPU is optional but recommended for training
- Organize your dataset as:
  dataset/
    train/
      class1/
      class2/
      ...
    val/
      class1/
      class2/
      ...
    test/  (optional)

## Quick start (local, VS Code)
1. Create & activate virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows (PowerShell/CMD)
   pip install -r requirements.txt
   ```
2. Add images into the `dataset/` folder as shown above. Each class folder should contain images.
3. Train the model:
   ```bash
   python train.py --data_dir dataset --epochs 10 --batch_size 16 --img_size 224
   ```
   The trained model will be saved to `models/rice_model.h5` and class indices to `models/class_indices.json`.
4. Run inference on a single image:
   ```bash
   python infer.py --model_path models/rice_model.h5 --image path/to/image.jpg
   ```
5. Run the web app (Flask):
   ```bash
   export FLASK_APP=app.py
   flask run --host=0.0.0.0 --port=5000
   ```
   Open `http://127.0.0.1:5000` and upload an image. The app will show predictions and a Grad-CAM overlay image.

## Files explained
- `train.py`: training script using MobileNetV2 transfer learning.
- `infer.py`: single-image inference and prints top-3 predictions.
- `app.py`: Flask app with upload UI + Grad-CAM endpoint.
- `gradcam.py`: helper to compute Grad-CAM heatmaps and overlay them.
- `utils.py`: preprocessing helpers.
- `requirements.txt`: recommended pip packages..

## Notes & troubleshooting
- If GPU usage is desired, install proper CUDA/CuDNN and matching TensorFlow GPU build.
- For low-resource devices, consider exporting to TensorFlow Lite (not included in this scaffold).
- This scaffold expects standard RGB JPEG/PNG images.

Good luck — add your dataset and start training locally!
