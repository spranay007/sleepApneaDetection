# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:25:50 2024

@author: spran
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JSONDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.data[idx]['ecg_data'], dtype=torch.float32).to(device)
        label = torch.tensor(1 if self.data[idx]['apnea_detected'] == 'A' else 0, dtype=torch.long).to(device)
        return feature, label

# Path to your JSON file
json_file = 'dataNew.json'

# Create custom dataset instance
dataset = JSONDataset(json_file)

# Split data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation data
batch_size = 64  # Example batch size
sequence_length = 6000 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class ECGCNN(nn.Module):
    def __init__(self, batch_size, sequence_length):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Calculate the output size of the convolutional and pooling layers
        conv_output_size = self._get_conv_output_size(batch_size, sequence_length)
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_conv_output_size(self, batch_size, sequence_length):
        # Dummy input tensor to get the output size of the convolutional layers
        x = torch.zeros((batch_size, 1, sequence_length))  # Assuming input size of (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return torch.flatten(x, 1).shape[1]

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output before the fully connected layers
        x = torch.flatten(x, 1)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model
num_time_points = dataset[0][0].shape[0]  # Assuming ECG data has fixed length
model = ECGCNN(batch_size, sequence_length).to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
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
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculating true positive, true negative, false positive, false negative
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            true_negative += ((predicted == 0) & (labels == 0)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')

# Evaluate the model
evaluate_model(model, val_loader)
