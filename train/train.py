import torch
import torch.nn as nn
import torch.optim as optim

from .utils import *

def train_model(model, train_loader, learning_rate, number_of_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = float('inf')

    for _ in range(number_of_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            try:
                model._detach()
            except:
                pass
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss

        if total_loss <= best_loss:
            best_loss = total_loss
            save_checkpoint(model, optimizer, number_of_epochs, best_loss)

def predict(model, test_loader):
    try:
        model, _, _, _ = load_checkpoint('checkpoint.pth', model)
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = torch.empty(0, dtype=torch.long)
    y_pred = torch.empty(0, dtype=torch.long) 

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)

            y_true = torch.cat((y_true, y_batch), dim=0)
            y_pred = torch.cat((y_pred, predictions), dim=0)
    
    return y_true, y_pred
    