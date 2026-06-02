import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ScribeIntel AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean(y_true * margin_square + (1 - y_true) * square_pred)

@tf.keras.utils.register_keras_serializable()
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        feats_a, feats_b = inputs
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(feats_a - feats_b), axis=1, keepdims=True))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

MODEL_PATH = os.path.join("models", "signature_siamese.keras")
if os.path.exists(MODEL_PATH):
    print("🎯 Loading native .keras Siamese Model context...")
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'contrastive_loss': contrastive_loss, 'DistanceLayer': DistanceLayer},
        safe_mode=False
    )
else:
    print("⚠️ Warning: Trained signature model weights file not found yet.")
    model = None

def preprocess_image(file_bytes: bytes, target_size=(105, 105)):
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("File stream could not be converted to a valid image matrix.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def read_root():
    return {"status": "online"}

@app.post("/api/analyze-handwriting")
async def analyze_handwriting(sample: UploadFile = File(...)):
    try:
        file_bytes = await sample.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # --- Enhanced OpenCV Structural Scanning Matrix ---
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mean_density = np.mean(thresh)
        
        # 1. Baseline & Line Trend Calculation
        row_sums = np.sum(thresh, axis=1)
        non_zero_rows = np.where(row_sums > (np.max(row_sums) * 0.05))[0]
        row_variance = np.var(non_zero_rows) if len(non_zero_rows) > 0 else 0
        
        # Calculate line gradient shift
        half_width = thresh.shape[1] // 2
        left_mass = np.mean(thresh[:, :half_width])
        right_mass = np.mean(thresh[:, half_width:])
        mass_trend = right_mass - left_mass

        # 2. Slant Calculations (Image Moments)
        moments = cv2.moments(thresh)
        mu11, mu02 = moments['mu11'], moments['mu02']
        slant_angle = 0.5 * np.arctan2(2 * mu11, (moments['mu20'] - mu02)) if (moments['mu20'] - mu02) != 0 else 0

        # 3. Word Spacing Factor
        col_sums = np.sum(thresh, axis=0)
        zero_cols = np.where(col_sums < (np.max(col_sums) * 0.02))[0]
        spacing_factor = len(zero_cols) / max(1, thresh.shape[1])

        metrics, traits = {}, []

        # --- Module 1: Slant Metric ---
        if slant_angle > 0.12:
            metrics['slant'] = "Right Slant (Dextrovert)"
            traits.extend(["Highly expressive persona", "Emotionally responsive", "Socially forward-moving"])
        elif slant_angle < -0.12:
            metrics['slant'] = "Left Slant (Sinistrovert)"
            traits.extend(["Reserved temperament", "Strong analytical self-protection", "Cautious emotional boundaries"])
        else:
            metrics['slant'] = "Vertical Alignment (Upright)"
            traits.extend(["Driven purely by logic", "Extremely independent", "Unbiased decision maker"])

        # --- Module 2: Spatial Baseline Metric ---
        if row_variance > 1800:
            metrics['baseline'] = "Fluctuating / Erratic Layout"
            traits.extend(["Highly creative fluid mindset", "Spontaneous reactions", "Unpredictable mood patterns"])
        else:
            metrics['baseline'] = "Straight / Rigid Baseline"
            traits.extend(["Disciplined lifestyle focus", "Reliable execution habit", "Goal-oriented precision"])

        # --- Module 3: Pen Pressure Metric ---
        if mean_density > 28:
            metrics['pressure'] = "Heavy Pen Pressure (High Vitality)"
            traits.extend(["Deep long-lasting emotions", "High sensory dedication", "Strong assertion dynamics"])
        else:
            metrics['pressure'] = "Light Pen Pressure (Rapid/Gentle)"
            traits.extend(["Quick cognitive adaptation", "Avoids heavy confrontations", "Highly sensitive nervous system"])

        # --- Module 4: Word Spacing ---
        if spacing_factor > 0.45:
            metrics['spacing'] = "Wide Word/Letter Gaps"
            traits.extend(["Desires social distance", "Values individual freedom", "Enjoys isolated deep work"])
        else:
            metrics['spacing'] = "Tight / Compressed Margins"
            traits.extend(["Seeks proximity and community", "Thrives in collaborative spaces", "Action-first execution cycle"])

        # --- Module 5: Line Trend ---
        if mass_trend > 1.5:
            metrics['trend'] = "Ascending Line Profile"
            traits.extend(["Ambitious mindset outlook", "Optimistic baseline expectations"])
        elif mass_trend < -1.5:
            metrics['trend'] = "Descending Line Profile"
            traits.extend(["Experiencing analytical fatigue", "Critical skeptical framework"])
        else:
            metrics['trend'] = "Balanced Linear Projection"
            traits.extend(["Emotionally stable state", "Steady work cadence"])

        # --- NEW Module 6: Automatic Gender Prediction Strategy ---
        # Graphological statistics link rounder, uniform spacing trends to higher feminine identifiers 
        # while sharper vertical/erratic baselines tilt towards masculine execution traits.
        if slant_angle > 0.05 and spacing_factor < 0.42:
            metrics['gender'] = "Female (Statistical Estimation)"
            traits.extend(["Strong emotional intelligence ties", "Detail integration affinity"])
        else:
            metrics['gender'] = "Male (Statistical Estimation)"
            traits.extend(["Highly structured physical output focus", "Independent problem-solving markers"])

        return {"metrics": metrics, "traits": list(set(traits))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal extraction fault: {str(e)}")

@app.post("/api/verify-signature")
async def verify_signature(genuine: UploadFile = File(...), suspect: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Signature neural weights file not found.")
    try:
        gen_bytes = await genuine.read()
        sus_bytes = await suspect.read()
        img_gen = preprocess_image(gen_bytes)
        img_sus = preprocess_image(sus_bytes)
        
        prediction = model.predict([img_gen, img_sus])
        distance = float(prediction[0][0])
        
        ACCEPTABLE_THRESHOLD = 0.30
        verdict = "GENUINE" if distance <= ACCEPTABLE_THRESHOLD else "FORGED / UNSURE"
        message = "Signature verified successfully." if verdict == "GENUINE" else "Forgery detected."

        return {"distance": distance, "verdict": verdict, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Execution Error: {str(e)}")