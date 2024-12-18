import torch.nn as nn
import torch

from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from typing import List, Union

from tqdm import tqdm

from sklearn.metrics import r2_score

import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(last_size, output_size),
        )
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100, learning_rate: Union[float, torch.Tensor] = 1e-3) -> None:
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_outputs = []
        all_train_labels = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
            for inputs, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                
                # Store outputs and labels for R² calculation
                all_train_outputs.append(outputs.detach().cpu().numpy())
                all_train_labels.append(labels.detach().cpu().numpy())
                
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(train_loss=train_loss / (tepoch.n + 1))

        train_loss /= len(train_loader)
        # Concatenate outputs and labels
        all_train_outputs = np.concatenate(all_train_outputs)
        all_train_labels = np.concatenate(all_train_labels)
        
        r2_train = r2_score(all_train_labels, all_train_outputs)

        model.eval()
        val_loss = 0
        all_val_outputs = []
        all_val_labels = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation', unit='batch') as vepoch:
                for inputs, labels in vepoch:
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Store outputs and labels for R² calculation
                    all_val_outputs.append(outputs.detach().cpu().numpy())
                    all_val_labels.append(labels.detach().cpu().numpy())
                    vepoch.set_postfix(val_loss=val_loss / (vepoch.n + 1))

        val_loss /= len(val_loader)
        # Concatenate outputs and labels
        all_val_outputs = np.concatenate(all_val_outputs)
        all_val_labels = np.concatenate(all_val_labels)
        
        r2_val = r2_score(all_val_labels, all_val_outputs)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, R²: {r2_train:.4f} - Validation Loss: {val_loss:.4f}, R²: {r2_val:.4f}')


def test(model: nn.Module, test_loader: DataLoader):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    all_test_outputs = []
    all_test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Store outputs and labels for R² calculation
            all_test_outputs.append(outputs.detach().cpu().numpy())
            all_test_labels.append(labels.detach().cpu().numpy())

    test_loss /= len(test_loader)
    # Concatenate outputs and labels
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    
    r2_test = r2_score(all_test_labels, all_test_outputs)

    print(f"Test Loss: {test_loss:.4f}, R²: {r2_test:.4f}")