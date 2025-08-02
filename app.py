from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5 * ssim_loss + 0.5 * l1_loss


brain_model = load_model('autoencoder8_brain_model.h5', custom_objects={'combined_loss': combined_loss})
kidney_model = load_model('autoencoder5_kidney_model.h5', custom_objects={'combined_loss': combined_loss})
brain_classifier = load_model('best_model1.h5')
kidney_classifier = load_model('tumor_classifier3_kidney.h5')  

brain_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
kidney_labels = ['no tumor', 'tumor']

def preprocess_image(img_path, target_size=(256, 256), normalize='-1to1'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    if normalize == '-1to1':
        img = img / 127.5 - 1
    elif normalize == '0to1':
        img = img / 255.0
    return np.expand_dims(img, axis=0)

def enhance_grayscale_sharpen(image):
    image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bright = clahe.apply(gray)
    bright = np.clip(bright * 1.15, 0, 255).astype(np.uint8)
    sharpened = cv2.filter2D(bright, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<organ>')
def upload_page(organ):
    return render_template('upload.html', organ=organ)

@app.route('/process/<organ>', methods=['POST'])
def process_image(organ):
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    original_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(original_path)


    model = brain_model if organ == 'brain' else kidney_model
    img_input = preprocess_image(original_path, target_size=(256, 256), normalize='-1to1')
    reconstructed = model.predict(img_input)
    enhanced = enhance_grayscale_sharpen(reconstructed[0])

    recon_filename = f'recon_{file.filename}'
    recon_path = os.path.join(UPLOAD_FOLDER, recon_filename)
    cv2.imwrite(recon_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

 
    if organ == 'brain':
        cls_input = preprocess_image(original_path, target_size=(240, 240), normalize='0to1')
        pred = brain_classifier.predict(cls_input)
        class_idx = np.argmax(pred)
        prediction = brain_labels[class_idx]
        confidence = float(pred[0][class_idx])
    elif organ == 'kidney':
        cls_input = preprocess_image(original_path, target_size=(256, 256), normalize='0to1')
        pred = kidney_classifier.predict(cls_input)
        class_idx = int(np.round(pred[0][0]))
        prediction = kidney_labels[class_idx]
        confidence = float(pred[0][0])
    else:
        return jsonify({'error': 'Invalid organ type'}), 400

    return jsonify({
        'original': url_for('static', filename=f'uploads/{file.filename}'),
        'denoised': url_for('static', filename=f'uploads/{recon_filename}'),
        'prediction': prediction,
        'confidence': f"{confidence:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
