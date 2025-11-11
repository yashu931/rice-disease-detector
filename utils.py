import cv2, numpy as np

def load_and_preprocess(path, target_size=(224,224)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError('Could not read image: ' + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32')/255.0
    return img
