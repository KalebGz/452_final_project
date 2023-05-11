import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss') 
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')  
    plt.legend()
    plt.grid()
    plt.show()


class BallDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        current_position = self.data.iloc[idx].values
        next_position = self.data.iloc[idx + 1].values
        return torch.FloatTensor(current_position), torch.FloatTensor(next_position)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs=100, patience=10):
    model.train()
    best_valid_loss = float('inf')
    no_improvement_epochs = 0

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        
        model.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)
        print(f"epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")


        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"early stopping at epoch {epoch + 1}")
            break
    return model, train_losses, valid_losses


def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss}")

    return test_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 2
    hidden_size = 128
    num_layers = 3
    batch_size = 4
    learning_rate = 0.01
    epochs = 300
    patience = 15

    dataset = BallDataset("ball_positions.csv")
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size,
    valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, train_losses, valid_losses = train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs, patience)
    plot_losses(train_losses, valid_losses)

    model.load_state_dict(torch.load("best_model.pth"))

    test_model(model, test_dataloader, criterion, device)


if __name__ == "__main__":
    main()
