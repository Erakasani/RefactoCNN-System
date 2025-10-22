
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from model.cnn_model import RefactoCNN

def objective(trial, X_train, y_train):
    input_dim = X_train.shape[1]
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
    
    model = RefactoCNN(input_dim=input_dim)
    model.dropout.p = dropout
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    acc = (predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
    return acc

def run_optimization(X_train, y_train):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    print("Best trial:")
    print(study.best_trial)
    return study.best_params
