import pandas as pd
import os

# Define file paths
mirai_files = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Mirai\archer-attack1.csv',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Mirai\camera-attack1.csv',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Mirai\indoor-attack1.csv'
}

normal_files = {
    'archer': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Normal\archer-normal.csv',
    'camera': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Normal\camera-normal.csv',
    'indoor': r'C:\Users\USER\IoT_Network_Traffic_Management\data\raw\Normal\indoor-normal.csv'
}

# Function to load, sample, and label datasets
def create_combined_dataset(device_name):
    # Load Normal data
    normal_data = pd.read_csv(normal_files[device_name])
    # Load Mirai data
    mirai_data = pd.read_csv(mirai_files[device_name])

    # Determine sample size (minimum of 50,000 or actual rows)
    normal_sample_size = min(50000, len(normal_data))
    mirai_sample_size = min(50000, len(mirai_data))

    # Sample the datasets
    normal_sampled = normal_data.sample(n=normal_sample_size, random_state=42)
    mirai_sampled = mirai_data.sample(n=mirai_sample_size, random_state=42)

    # Label the datasets
    normal_sampled['label'] = 0  # Normal = 0
    mirai_sampled['label'] = 1    # Mirai = 1

    # Combine the datasets
    combined_dataset = pd.concat([normal_sampled, mirai_sampled], ignore_index=True)

    # Shuffle the combined dataset
    combined_dataset = combined_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined_dataset

# Create combined datasets for Archer, Camera, and Indoor
devices = ['archer', 'camera', 'indoor']
combined_datasets = {}

for device in devices:
    combined_datasets[device] = create_combined_dataset(device)
    # Save to CSV
    combined_datasets[device].to_csv(f'C:\\Users\\USER\\IoT_Network_Traffic_Management\\data\\processed\\combined_{device}.csv', index=False)

print("Combined datasets created for Archer, Camera, and Indoor devices.")
