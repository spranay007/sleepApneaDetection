# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:49:36 2024

@author: spran
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming you have your ECG data and labels loaded into numpy arrays
# X contains ECG data (shape: [num_samples, num_time_points])
# y contains corresponding labels (0 or 1)
import json
import torch
from torch.utils.data import Dataset, DataLoader

class JSONDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            dataAll = json.load(f)
        self.data = dataAll
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.data[idx]['ecg_data'], dtype=torch.float32)
        label = torch.tensor(1 if self.data[idx]['apnea_detected'] == 'A' else 0, dtype=torch.long)  # Assuming labels are integers
        return feature, label

# Path to your JSON file
json_file = 'dataNew.json'

# Create custom dataset instance
dataset = JSONDataset(json_file)

# Create DataLoader
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage:
for batch in data_loader:
    features, labels = batch
print(features,labels)    

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.clone().detach(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.clone().detach(), dtype=torch.long)  # Assuming class labels are integers
X_val_tensor = torch.tensor(X_val.clone().detach(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.clone().detach(), dtype=torch.long)

# Create DataLoader for training and validation data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(X_train_tensor.shape)
# Define the CNN model
class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * ((num_time_points - 4) // 2 - 4) , 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * ((num_time_points - 4) // 2 - 4))  # Flatten before FC layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model
num_time_points = X_train.shape[1]  # Assuming ECG data has fixed length
model = ECGCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluation
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')

# Evaluate the model
evaluate_model(model, val_loader)
