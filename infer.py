import argparse, json, os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_class_indices(path='models/class_indices.json'):
    with open(path,'r') as f:
        d = json.load(f)
    # invert map: idx->class
    return {v:k for k,v in d.items()}

def predict(model_path, img_path, top_k=3, target_size=(224,224)):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x,0)
    preds = model.predict(x)[0]
    idx_to_class = load_class_indices()
    sorted_idx = preds.argsort()[::-1]
    results = []
    for i in sorted_idx[:top_k]:
        results.append((idx_to_class[i], float(preds[i])))
    return results

if __name__ == '__main__':
    import os, random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/rice_model.h5')
    parser.add_argument('--image', default=None, help='Path to test image (optional)')
    parser.add_argument('--random', action='store_true', help='Pick a random image from dataset/val/')
    args = parser.parse_args()

    if args.random:
        val_dir = 'dataset/val'
        cls = random.choice(os.listdir(val_dir))
        cls_path = os.path.join(val_dir, cls)
        img = random.choice([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        args.image = os.path.join(cls_path, img)
        print(f"🔍 Using random image: {args.image}")

    if not args.image or not os.path.exists(args.image):
        raise FileNotFoundError(f"Image path invalid or missing: {args.image}")

    res = predict(args.model_path, args.image)
    print('Top predictions:')
    for cls, score in res:
        print(f'{cls}: {score:.4f}')

