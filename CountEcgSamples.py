import json

def count_ecg_minute_data(json_data):
    ecg_counts = {}
    for entry in json_data:
        patient_name = entry["patientName"]
        ecg_data_count = len(entry["ecg_data"])
        if patient_name in ecg_counts:
            ecg_counts[patient_name].append((entry["chunkid"], ecg_data_count))
        else:
            ecg_counts[patient_name] = [(entry["chunkid"], ecg_data_count)]
    return ecg_counts

def find_max_min_counts(ecg_counts):
    max_min_counts = {}
    for patient_name, counts in ecg_counts.items():
        min_count = min(counts, key=lambda x: x[1])[1]  # Find the minimum count
        min_chunks = [chunk_id for chunk_id, count in counts if count == min_count]  # Find chunk IDs with the minimum count
        max_min_counts[patient_name] = {"min_count": min_count, "min_chunks": min_chunks}
    return max_min_counts

# Read JSON file
json_file_path = "dataNew.json"  # Change this to the path of your JSON file
with open(json_file_path, "r") as file:
    json_data = json.load(file)

# Count ECG minute data for each patient
ecg_counts = count_ecg_minute_data(json_data)

# Find minimum counts and corresponding chunk IDs for each patient
max_min_counts = find_max_min_counts(ecg_counts)

# Print results
for patient_name, counts_info in max_min_counts.items():
    min_count = counts_info["min_count"]
    min_chunks = counts_info["min_chunks"]
    print(f"Patient: {patient_name}, Minimum ECG minute data count: {min_count}, Chunk IDs with minimum count: {min_chunks}")
