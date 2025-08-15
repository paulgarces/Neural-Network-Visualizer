# setting up self tuning neural network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the California housing dataset
cali_data = fetch_california_housing()
cali_df = pd.DataFrame(cali_data==cali_data.data, columns=cali_data.feature_names)
cali_df["target"] = cali_data.target
# print(cali_df.head(5))

# getting the features and target
X = cali_df.drop("target", axis=1).values
y = cali_df["target"].values.reshape(-1, 1)

# scaling the features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=78)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Neural Network Class
class SelfTuningNN(nn.Module):
    def __init__(self, input_size, hidden_size = 64):
        super(SelfTuningNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

input_size = X_train.shape[1]
model = SelfTuningNN(input_size = input_size)

error_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    model.train()
    predictions = model(X_train)
    loss = error_criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        model.eval()
        val_predictions = model(X_test)
        val_loss = error_criterion(val_predictions, y_test)
        print(f"Epoch {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")