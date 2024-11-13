from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Paths to the saved models and label encoders
model_paths = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Archer_model.h5',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Camera_model.h5',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Indoor_model.h5'
}

label_encoder_paths = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\models\label\label_encoder_archer.pkl',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\models\label\label_encoder_camera.pkl',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\models\label\label_encoder_indoor.pkl'
}

# Load models and label encoders
models = {device: load_model(model_paths[device]) for device in model_paths}
label_encoders = {device: joblib.load(label_encoder_paths[device]) for device in label_encoder_paths}

# StandardScaler for preprocessing
scalers = {device: StandardScaler() for device in model_paths}

def preprocess_data(device, input_data):
    # Standardize features
    X = np.array(input_data).reshape(1, -1)  # Reshape for scaler
    X = scalers[device].fit_transform(X)  # Standardize
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for CNN-LSTM
    return X

@app.route('/predict/<device>', methods=['POST'])
def predict(device):
    if device not in models:
        return jsonify({'error': 'Invalid device name'}), 400
    
    # Get input data from request
    input_data = request.json.get('data')
    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400
    
    # Preprocess the input data
    try:
        X_test = preprocess_data(device, input_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make predictions
    predictions = models[device].predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Decode labels
    decoded_labels = label_encoders[device].inverse_transform(predicted_classes)

    return jsonify({'predictions': decoded_labels.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
