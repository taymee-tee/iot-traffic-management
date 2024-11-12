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
