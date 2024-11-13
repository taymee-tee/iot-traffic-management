from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Paths to model and label encoder files
model_paths = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Archer_model.h5',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Camera_model.h5',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\models\IOT_Indoor_model.h5'
}
label_encoder_paths = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\label\label_encoder_archer.pkl',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\label\label_encoder_camera.pkl',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\label\label_encoder_indoor.pkl'
}

# Load models and encoders
models = {device: load_model(path) for device, path in model_paths.items()}
label_encoders = {device: joblib.load(path) for device, path in label_encoder_paths.items()}

# Endpoint to test model prediction with provided data
@app.route('/predict/<device>', methods=['POST'])
def predict(device):
    if device not in models:
        return jsonify({'error': 'Device model not found'}), 400
    
    data = request.json.get('data')
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Preprocess data
        X = np.array(data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        # Make predictions
        model = models[device]
        y_pred = model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Decode predicted labels
        label_encoder = label_encoders[device]
        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

        return jsonify({'predictions': y_pred_labels.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to get model information
@app.route('/model_info/<device>', methods=['GET'])
def model_info(device):
    if device not in models:
        return jsonify({'error': 'Device model not found'}), 400

    return jsonify({
        'device': device,
        'input_shape': models[device].input_shape,
        'output_shape': models[device].output_shape
    })

if __name__ == '__main__':
    app.run(debug=True)
