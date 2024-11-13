# IoT Network Traffic Management

## Project Overview

This project focuses on enhancing the security and efficiency of IoT networks using network traffic simulation, analysis, and AI-driven predictions. By simulating IoT devices and analyzing network traffic, it helps network administrators better manage bandwidth, detect anomalies, and secure their IoT environments.

**Key Features**:
- **IoT Network Simulation** with GNS3
- **Traffic Generation** using Iperf3
- **Network Traffic Capture and Monitoring** via Wireshark
- **AI Traffic Classification** using pre-trained models (CNN & LSTM)
- **Real-time Traffic Analysis** using Flask API

## Technologies Used
- **Network Simulation**: GNS3
- **Traffic Generation**: Iperf3
- **Traffic Analysis**: Wireshark
- **AI & ML**: TensorFlow, Keras
- **Backend**: Flask (for real-time predictions)
- **Data Science Libraries**: NumPy, pandas, scikit-learn

## Requirements

### Hardware:
- **CPU**: Quad-core or higher
- **RAM**: 16 GB minimum
- **Disk Space**: 20 GB free space
- **Network**: Ethernet/Wi-Fi Adapter, Microsoft Loopback Adapter

### Software:
- **GNS3**: For IoT network simulation
- **Iperf3**: For generating network traffic
- **Wireshark**: For capturing and analyzing network traffic
- **Anaconda**: Python environment for ML model training
- **Python Libraries**: TensorFlow, Keras, pandas, numpy, scikit-learn, Flask

### Install the Required Software

1. **Install GNS3** from [here](https://www.gns3.com/).
2. **Download Iperf3** from [iperf.fr](https://iperf.fr/).
3. **Install Wireshark** from [Wireshark.org](https://www.wireshark.org/).
4. **Install Anaconda** from [Anaconda.com](https://www.anaconda.com/products/individual).

### Install Python Dependencies
After setting up Anaconda, create a new environment and install the required Python packages:

```bash
conda create -n iot_env python=3.8
conda activate iot_env
pip install -r requirements.txt
```

# Project Setup

## 1. Clone the Repository
Clone the project repository to your local machine:

```bash
git clone https://github.com/taymee-tee/iot-traffic-management.git
cd iot-traffic-management
```
## 2. Set Up Network Simulation (GNS3)
- Launch GNS3 and create a new project.
- Add devices like routers, cameras, and sensors to the workspace.
- Configure static IPs for devices using the GNS3 console.

## 3. Generate Traffic (Iperf3)
Start the Iperf3 server on the router:

```bash

iperf3 -s -p 5201
```
Generate traffic from IoT devices (for example, a camera):

```bash
iperf3 -c 192.168.1.1 -p 5201 -u -b 2M -t 120
```
## 4. Capture Traffic (Wireshark)
- Open Wireshark and start capturing packets.
- Filter traffic for a specific IP address (e.g., ip.addr == 192.168.1.2).

  
## 5. Run AI Model for Traffic Classification
Once the environment is set up and traffic is generated, use the AI models for classification.

Launch Jupyter Notebook to run model training and analysis:

```bash
jupyter notebook
```
## 6. Flask API for Real-time Predictions
Run the Flask API to receive real-time traffic predictions:

```bash
cd src
python app.py
```
Send traffic data to the API for classification:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @traffic_data.json
```
