from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import random
import time

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

# Load models and encoders with simulated loading
models = {}
label_encoders = {}

print("Loading models and encoders...")
for device, path in model_paths.items():
    print(f"Loading model for {device}...")
    time.sleep(1)  # Simulate model loading time
    models[device] = load_model(path)
    label_encoders[device] = joblib.load(label_encoder_paths[device])
    print(f"Model for {device} loaded successfully.")

# Paths to traffic data
data_paths = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\processed_archer.csv',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\processed_camera.csv',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\data\processed_data\processed_indoor.csv'
}

# Load the datasets
datasets = {device: pd.read_csv(path) for device, path in data_paths.items()}

# Function to get a random sample from the dataset
def get_random_sample(device):
    dataset = datasets.get(device)
    if dataset is None:
        return None
    random_row = dataset.sample(n=1)
    return random_row.iloc[:, :7].values  # Select only the first 7 features

# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Real API Endpoint to test model prediction with provided data
@app.route('/predict/<device>', methods=['POST'])
def predict(device):
    if device not in models:
        return jsonify({'error': 'Device model not found'}), 400

    data = request.json.get('data')
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    try:
        X = np.array(data)

        # Ensure input data has exactly 7 features
        if X.shape[1] < 7:
            padding = np.zeros((X.shape[0], 7 - X.shape[1]))
            X = np.hstack((X, padding))
        elif X.shape[1] > 7:
            X = X[:, :7]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        model = models[device]
        y_pred = model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        label_encoder = label_encoders.get(device)
        if label_encoder is None:
            return jsonify({'error': f'Label encoder for {device} not found'}), 400

        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
        return jsonify({'predictions': y_pred_labels.tolist()})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Deceptive API Endpoint that returns random predictions
@app.route('/deceptive_predict/<device>', methods=['POST'])
def deceptive_predict(device):
    print(f"Running deceptive prediction for device: {device}...")
    time.sleep(1)  # Simulate processing time

    # Generate random "fake" predictions
    normal_traffic = ["Normal", "Low Bandwidth Usage", "No Threat Detected"]
    abnormal_traffic = ["High Bandwidth Usage", "Potential Threat Detected", "DDoS Attack"]
    is_abnormal = random.randint(1, 30) == 1  # 1 in 30 chance of abnormal
    predictions = [random.choice(abnormal_traffic)] if is_abnormal else [random.choice(normal_traffic)]

    return jsonify({"predictions": predictions})

# Endpoint to get live traffic data for random samples
@app.route('/live_traffic/<device>', methods=['GET'])
def live_traffic(device):
    sample = get_random_sample(device)
    if sample is None:
        return jsonify({'error': 'No data available for this device'}), 400
    return jsonify({'data': sample.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
