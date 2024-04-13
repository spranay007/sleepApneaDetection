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

# Prompt user for patient information
patient_id = input("Enter patient ID: ")
sex = input("Enter patient sex (M/F): ")
age = input("Enter patient age: ")

# Create JSON data
patient_data = []
sample_rate_ecg = record_ecg[1]['fs']
samples_per_minute = 60 * sample_rate_ecg  # Number of samples in one minute
for i in range(0, len(annotation.symbol)):
    apnea_minute_data = annotation.symbol[i]
    if apnea_minute_data is None:  # Check if apnea_minute_data is null
        print("Null apnea value encountered. Stopping JSON creation.")
        patient_data = []
        break
    ecg_minute_data = ecg_data[i * samples_per_minute: (i + 1) * samples_per_minute].tolist()
    minute_data = {
        "ecg_data": ecg_minute_data,
        "apnea_detected": apnea_minute_data
    }
    patient_data.append(minute_data)

if patient_data:  # Check if patient_data is not empty
    # Create patient information dictionary
    patient_info = {
        "patient_id": patient_id,
        "sex": sex,
        "age": age
    }

    # Combine patient information and data
    json_data = {"patient_data": patient_data}

    # Write JSON to file
    output_file = "dataNew.json"
    with open(output_file, "w") as json_file:
        json.dump(patient_data, json_file, indent=2)

    print("JSON file created successfully.")
