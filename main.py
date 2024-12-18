import os

from models.mlp import MLP
from data.utils import create_dataloaders_from_dataset_path, get_data_shape

dataset = os.path.join(os.getcwd(), "data", "dataset.csv")

train_loader, val_loader, test_loader = create_dataloaders_from_dataset_path(dataset)

input_size = get_data_shape()
hidden_sizes = [64, 128, 128, 128, 64]
output_size = 1

mlp = MLP()