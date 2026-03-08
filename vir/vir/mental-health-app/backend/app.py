from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import cv2
import librosa
import base64
from datetime import datetime
import jwt
import bcrypt
from functools import wraps
import torch


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'


# -----------------------------
# Load ML Models (with CPU-safe text model)
# -----------------------------
def load_models():
    audio_model = None
    video_model = None
    text_model = None

    # Audio model (may fail due to TF version)
    try:
        audio_model = tf.keras.models.load_model('audio_model.h5', compile=False)
        print("Audio model loaded.")
    except Exception as e:
        print(f"Warning: Could not load audio model - {e}")
        audio_model = None

    # Video model
    try:
        video_model = tf.keras.models.load_model('video_model.h5', compile=False)
        print("Video model loaded.")
    except Exception as e:
        print(f"Warning: Could not load video model - {e}")
        video_model = None

    # Text model (PyTorch, force CPU)
    try:
        import torch
        text_model = torch.load('text_model.pkl', map_location=torch.device('cpu'))
        
        print("Text model loaded on CPU.")
    except Exception as e:
        print(f"Warning: Could not load text model - {e}")
        text_model = None

    return audio_model, video_model, text_model


audio_model, video_model, text_model = load_models()


# -----------------------------
# Mock Database
# -----------------------------
users_db = {}


# -----------------------------
# Auth Decorator
# -----------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token missing'}), 401

        try:
            token = token.split()[1]  # Remove "Bearer"
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user']
        except:
            return jsonify({'message': 'Invalid token'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# -----------------------------
# Register
# -----------------------------
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if username in users_db:
        return jsonify({'message': 'User already exists'}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    users_db[username] = {
        'password': hashed,
        'email': email,
        'created_at': datetime.now().isoformat()
    }

    return jsonify({'message': 'Registration successful'}), 201


# -----------------------------
# Login
# -----------------------------
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username not in users_db:
        return jsonify({'message': 'Invalid credentials'}), 401

    if bcrypt.checkpw(password.encode('utf-8'), users_db[username]['password']):
        token = jwt.encode({
            'user': username,
            'exp': datetime.utcnow().timestamp() + 86400
        }, app.config['SECRET_KEY'], algorithm="HS256")

        return jsonify({'token': token, 'username': username}), 200

    return jsonify({'message': 'Invalid credentials'}), 401


# -----------------------------
# Audio Processing
# -----------------------------
def process_audio(audio_data):
    try:
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        mfccs = np.mean(audio_array.reshape(-1, 128)[:, :40], axis=0)
        return mfccs.reshape(1, -1)
    except Exception as e:
        print(f"Audio processing error: {e}")
        return None


# -----------------------------
# Video Processing
def process_video(video_frame):
    """Process video frame and return features for model prediction"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(video_frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)

        # Load as GRAYSCALE (1 channel)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize to 48x48 as expected by the model
        img_resized = cv2.resize(img, (48, 48))

        # Normalize to [0, 1]
        img_normalized = img_resized.astype("float32") / 255.0

        # Add channel dimension: (48, 48) -> (48, 48, 1)
        img_normalized = np.expand_dims(img_normalized, axis=-1)

        # Add batch dimension: (48, 48, 1) -> (1, 48, 48, 1)
        return np.expand_dims(img_normalized, axis=0)

    except Exception as e:
        print(f"Video processing error: {e}")
        return None

# -----------------------------
# Text Processing
# -----------------------------
def process_text(text):
    try:
        if text_model and hasattr(text_model, 'transform'):
            return text_model.transform([text])
        return None
    except Exception as e:
        print(f"Text processing error: {e}")
        return None


# -----------------------------
# Analyze Mental Health
# -----------------------------
@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze(current_user):
    data = request.json

    audio_data = data.get('audio')
    video_frame = data.get('video')
    text_data = data.get('text')

    scores = {}

    # Audio
    if audio_data and audio_model:
        audio_features = process_audio(audio_data)
        if audio_features is not None:
            pred = audio_model.predict(audio_features)
            scores['audio_score'] = float(pred[0][0])

    # Video
    if video_frame and video_model:
        video_features = process_video(video_frame)
        if video_features is not None:
            pred = video_model.predict(video_features)
            scores['video_score'] = float(pred[0][0])

    # Text
    if text_data and text_model:
        text_features = process_text(text_data)
        if text_features is not None:
            if hasattr(text_model, 'predict_proba'):
                pred = text_model.predict_proba(text_features)
                scores['text_score'] = float(pred[0][1])
            else:
                pred = text_model.predict(text_features)
                scores['text_score'] = float(pred[0])

    # Combined Score
    valid = [v for v in scores.values() if v is not None]
    combined_score = sum(valid) / len(valid) if valid else 0

    # Risk Level
    if combined_score < 0.3:
        risk = "Low"
        feedback = "Your responses indicate positive well-being."
    elif combined_score < 0.6:
        risk = "Moderate"
        feedback = "Some indicators suggest mild concerns."
    else:
        risk = "High"
        feedback = "Strongly recommended to talk to a professional."

    return jsonify({
        'scores': scores,
        'combined_score': combined_score,
        'risk_level': risk,
        'feedback': feedback,
        'timestamp': datetime.now().isoformat()
    })


# -----------------------------
# Chatbot
# -----------------------------
@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    questions = [
        "How have you been feeling lately?",
        "Have you experienced changes in your sleep?",
        "Do you feel anxious often?",
        "How is your energy level today?",
        "What has been stressing you recently?"
    ]

    return jsonify({
        'message': np.random.choice(questions),
        'timestamp': datetime.now().isoformat()
    })


# -----------------------------
# Health Check
# -----------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'audio': audio_model is not None,
            'video': video_model is not None,
            'text': text_model is not None
        }
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
