# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 03:45:34 2024

@author: spran
"""

import json

def count_ecg_minute_data(json_data):
    ecg_counts = {}
    for entry in json_data:
        patient_name = entry["patientName"]
        ecg_data_count = len(entry["ecg_data"])
        if patient_name in ecg_counts:
            ecg_counts[patient_name].append(ecg_data_count)
        else:
            ecg_counts[patient_name] = [ecg_data_count]
    return ecg_counts

def find_max_min_counts(ecg_counts):
    max_count = float('-inf')
    min_count = float('inf')
    max_patient = ""
    min_patient = ""
    for patient_name, counts in ecg_counts.items():
        curr_max = max(counts)
        curr_min = min(counts)
        if curr_max > max_count:
            max_count = curr_max
            max_patient = patient_name
        if curr_min < min_count:
            min_count = curr_min
            min_patient = patient_name
    return max_count, max_patient, min_count, min_patient

# Read JSON file
json_file_path = "dataNew.json"  # Change this to the path of your JSON file
with open(json_file_path, "r") as file:
    json_data = json.load(file)

# Count ECG minute data for each patient
ecg_counts = count_ecg_minute_data(json_data)

# Find maximum and minimum counts along with patient names
max_count, max_patient, min_count, min_patient = find_max_min_counts(ecg_counts)

# Print results
print("Maximum ECG minute data count:", max_count, "for patient:", max_patient)
print("Minimum ECG minute data count:", min_count, "for patient:", min_patient)

