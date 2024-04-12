# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:44:46 2024

@author: spran
"""

import os
import wfdb
import numpy as np
import json

# Read ECG data
recordname_ecg = "./apnea-ecg-database/setOne/a01"
record_ecg = wfdb.rdsamp(recordname_ecg)
ecg_data = record_ecg[0][:, 0]  # Selecting the first channel's data

# Read apnea notation
recordname_apnea = "./apnea-ecg-database/setOne/a01r"
if os.path.isfile(recordname_apnea + '.apn'):  # Checking if the apn file exists
    annotation = wfdb.rdann(recordname_apnea, extension="apn")
else:
    print("Apnea annotation file not found!")
    exit()

# Create JSON data
patient_data = []
sample_rate_ecg = record_ecg[1]['fs']
samples_per_minute = 60 * sample_rate_ecg  # Number of samples in one minute
for i in range(0, len(ecg_data), samples_per_minute):
    ecg_minute_data = ecg_data[i: i + samples_per_minute].tolist()
    apnea_minute_index = int(i / samples_per_minute)
    if apnea_minute_index < len(annotation.symbol):  # Check if apnea_minute_index is within the range of annotation.symbol
        apnea_minute_data = annotation.symbol[apnea_minute_index]  # Take one sample from the apnea file per minute
    else:
        apnea_minute_data = None  # Set to None if index is out of range
    minute_data = {
        "ecg_data": ecg_minute_data,
        "apnea_detected": apnea_minute_data
    }
    patient_data.append(minute_data)

json_data = {"patient_data": patient_data}

# Write JSON to file
output_file = "data.json"
with open(output_file, "w") as json_file:
    json.dump(json_data, json_file, indent=2)

print("JSON file created successfully.")
