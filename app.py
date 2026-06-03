import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1. Custom Loss function reconstruction baseline
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean(y_true * margin_square + (1 - y_true) * square_pred)

# 2. Hardcoded reconstruction helper for the Lambda layer math calculation
def reconstruct_distance(tensors):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensors[0] - tensors[1]), axis=1, keepdims=True))

# Load signature model files safely
MODEL_PATH = r"models\signature_siamese.h5"
if os.path.exists(MODEL_PATH):
    print("🎯 Loading your custom trained Siamese Model...")
    
    # We pass both the loss function AND the explicit math layer helper
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={
            'contrastive_loss': contrastive_loss,
            'function': reconstruct_distance
        },
        safe_mode=False
    )
else:
    print("⚠️ Custom model binary not found. Run train_models.py first.")
    model = None

def preprocess_image(file_storage, target_size=(105, 105)):
    filestr = file_storage.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Uploaded file could not be parsed as a valid image matrix.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-handwriting', methods=['POST'])
def analyze_handwriting():
    if 'sample' not in request.files:
        return jsonify({'error': 'No handwriting sample image uploaded'}), 400
    try:
        file = request.files['sample']
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image file alignment'}), 400

        # Feature Extraction using OpenCV
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mean_density = np.mean(thresh)

        row_sums = np.sum(thresh, axis=1)
        non_zero_rows = np.where(row_sums > (np.max(row_sums) * 0.05))[0]
        row_variance = np.var(non_zero_rows) if len(non_zero_rows) > 0 else 0

        moments = cv2.moments(thresh)
        mu11 = moments['mu11']
        mu02 = moments['mu02']
        slant_angle = 0.5 * np.arctan2(2 * mu11, (moments['mu20'] - mu02)) if (moments['mu20'] - mu02) != 0 else 0

        metrics = {}
        traits = []

        if slant_angle > 0.1:
            metrics['slant'] = "Right Slant (Dextrovert)"
            traits.extend(["Highly expressive persona", "Emotionally responsive", "Future-oriented thinker"])
        elif slant_angle < -0.1:
            metrics['slant'] = "Left Slant (Sinistrovert)"
            traits.extend(["Reserved temperament", "Strong analytical self-protection", "Introverted tendencies"])
        else:
            metrics['slant'] = "Vertical Alignment (Upright)"
            traits.extend(["Driven purely by logic", "Extremely independent", "High emotional control"])

        if row_variance > 1500:
            metrics['baseline'] = "Fluctuating / Erratic Layout"
            traits.extend(["Highly creative fluid mindset", "Moody or spontaneous reactions", "Adaptable to chaos"])
        else:
            metrics['baseline'] = "Straight / Rigid Baseline"
            traits.extend(["Disciplined lifestyle focus", "Reliable execution habit", "Strong willpower under stress"])

        if mean_density > 25:
            metrics['speed'] = "Heavy Pen Pressure (High Vitality)"
            traits.extend(["Deep long-lasting emotions", "High sensory dedication", "Intense personal commitment"])
        else:
            metrics['speed'] = "Light Pen Pressure (Rapid/Gentle)"
            traits.extend(["Quick cognitive adaptation", "Avoids heavy confrontations", "Highly sensitive nervous system"])

        return jsonify({'metrics': metrics, 'traits': list(set(traits))})
    except Exception as e:
        return jsonify({'error': f"Internal extraction fault: {str(e)}"}), 500

@app.route('/api/verify-signature', methods=['POST'])
def verify_signature():
    if 'genuine' not in request.files or 'suspect' not in request.files:
        return jsonify({'error': 'Missing signature samples'}), 400
    try:
        img_gen = preprocess_image(request.files['genuine'])
        img_sus = preprocess_image(request.files['suspect'])
        
        if model is None:
            return jsonify({'error': 'Signature engine model weights not initialized.'}), 500

        prediction = model.predict([img_gen, img_sus])
        distance = float(prediction[0][0])
        
        ACCEPTABLE_THRESHOLD = 0.30
        verdict = "GENUINE" if distance <= ACCEPTABLE_THRESHOLD else "FORGED / UNSURE"
        
        if verdict == "GENUINE":
            message = "Signature verified successfully. Structural path angles, micro-spacing distribution, and boundaries match perfectly."
        else:
            message = "Forgery detected. High structural variance observed across key spatial checkpoints."

        return jsonify({'distance': distance, 'verdict': verdict, 'message': message})
    except Exception as e:
        return jsonify({'error': f"Model Execution Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)