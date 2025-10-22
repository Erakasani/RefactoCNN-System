
import torch
import torch.nn as nn
import torch.optim as optim
from model.cnn_model import RefactoCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def train_model(X, y, input_dim=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RefactoCNN(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Evaluate
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    print(classification_report(y_test, predicted.numpy()))
    
    return model
