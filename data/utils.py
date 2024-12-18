import torch
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def create_dataloaders_from_dataset_path(file_path: Path):
    df = pd.read_csv(file_path)
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values
    y = df['PowerConsumption'].values

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader

def get_data_shape(file_path: Path):

    df = pd.read_csv(file_path)
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values

    return X.shape[1]

def get_val_samples(file_path:Path, num_samples:int=100):
    df = pd.read_csv(file_path)
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values

    return torch.tensor(X[:num_samples], dtype=torch.float32)