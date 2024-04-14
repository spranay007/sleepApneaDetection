import os
import wfdb
import numpy as np
import json

def process_files(directory):
    json_data = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.dat'):  # Check if it's an ECG file
            ecg_path = os.path.join(directory, filename)
            apnea_path = os.path.join(directory, filename[:-4] + '.apn')

            # Check if corresponding apnea file exists
            if not os.path.exists(apnea_path):
                print(f"Apnea annotation file not found for {filename}!")
                continue

            try:
                # Read ECG data
                record_ecg = wfdb.rdsamp(ecg_path[:-4])
                ecg_data = record_ecg[0][:, 0]  # Selecting the first channel's data

                # Read apnea notation
                annotation = wfdb.rdann(apnea_path[:-4], extension="apn")

                # Create JSON data
                sample_rate_ecg = 100  # Assuming sampling rate is 100 samples per second
                samples_per_minute = sample_rate_ecg * 60  # Number of samples in one minute
                for i in range(0, len(annotation.symbol)):
                    apnea_minute_data = annotation.symbol[i]
                    ecg_minute_data = ecg_data[i * samples_per_minute: (i + 1) * samples_per_minute].tolist()

                    # Ensure ECG data count is exactly 6000
                    if len(ecg_minute_data) == 6000:
                        minute_data = {
                            "patientName": os.path.splitext(filename)[0],  # Extracting patient name without extension
                            "chunkid": i,
                            "ecg_data": ecg_minute_data,
                            "apnea_detected": apnea_minute_data
                        }
                        json_data.append(minute_data)
                    else:
                        print(f"Skipping incomplete minute for {filename}, Chunk ID: {i}")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    return json_data

# Directory where ECG and apnea files are stored
directory = "./apnea-ecg-database/ECGTrainingData"
json_data = process_files(directory)

# Write JSON to file
output_file = "dataNew.json"
with open(output_file, "w") as json_file:
    json.dump(json_data, json_file, indent=2)

print("JSON file created successfully.")
