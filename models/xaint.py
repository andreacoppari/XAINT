import torch
import shap

import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from typing import List

from torchviz import make_dot

import tempfile
import os

from data.utils import get_val_samples
from models.mlp import MLP


class XAINT(nn.Module):
    """
    XAINT (eXplainable Ai through Iterative Network Truncation) is a PyTorch neural network module that creates a truncated version of a given MLP model.
    It retains the first k layers from the original model and attaches an adjustment layer to align 
    the output dimensions with the classifier layer of the original model.

    Parameters:
    ----------
    model : MLP
        The original MLP model from which to truncate the layers.
    k : int
        The number of layers to remove from the original model's hidden layers. The layers are truncated
        from the end of the hidden layers.

    Attributes:
    ----------
    truncated_model : nn.Sequential
        A sequential container of the first k hidden layers from the original model.
    classifier : nn.Sequential
        The classifier layer from the original model.
    adjust_layer : nn.Linear or nn.Identity
        A linear layer that adjusts the output size from the truncated model to match the input size
        of the classifier. If the sizes are already compatible, this is an identity layer.

    Methods:
    -------
    forward(x: Tensor) -> Tensor
        Performs a forward pass of the input tensor through the truncated model, adjustment layer,
        and classifier.

    Notes:
    -----
    - If the number of layers to be kept (k) is such that the dimensions between the truncated output
      and the classifier input are different, an adjustment layer is added and initialized using
      Xavier uniform initialization.
    - Weights for both the truncated model and classifier are copied from the original model to ensure
      consistency.
    """
    def __init__(self, model: MLP, k: int):
        """
        Initializes the XAINT class by truncating the given MLP model and setting up the necessary
        adjustment layer.

        Parameters:
        ----------
        model : MLP
            The original MLP model to truncate.
        k : int
            The number of layers to remove from the original model's hidden layers.
        """
        super(XAINT, self).__init__()

        # number of layers to keep
        k = len(model.hidden_layers)-k*2

        self.truncated_model = nn.Sequential(*list(model.hidden_layers.children())[:k])
        self.classifier = model.classifier

        truncated_output_size = list(model.hidden_layers.children())[k - 2].out_features
        classifier_input_size = model.classifier[0].in_features
        
        if truncated_output_size != classifier_input_size:
            self.adjust_layer = nn.Linear(truncated_output_size, classifier_input_size)
            init.xavier_uniform_(self.adjust_layer.weight)
        else:
            self.adjust_layer = nn.Identity()

        # Load weights for truncated_model
        for i, layer in enumerate(self.truncated_model):
            if isinstance(layer, nn.Linear):
                original_layer = model.hidden_layers[i]
                layer.weight.data = original_layer.weight.data.clone()
                layer.bias.data = original_layer.bias.data.clone()

        # Load weights for classifier
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                classifier_index = i
                original_layer = model.classifier[classifier_index]
                layer.weight.data = original_layer.weight.data.clone()
                layer.bias.data = original_layer.bias.data.clone()


    def forward(self, x):
        x = self.truncated_model(x)
        x = self.adjust_layer(x)
        x = self.classifier(x)
        return x


    def test(self, test_loader: DataLoader):
        self.eval()
        criterion = nn.MSELoss()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")


    def shap_explain(self, dataset, num_samples: int = 64):
        X_samples = get_val_samples(dataset, num_samples)
        explainer = shap.DeepExplainer(self, X_samples)
        shap_values = explainer.shap_values(X_samples)
        
        shap.initjs()
        
        def plot_shap_bar(shap_values, feature_names):
            shap_values = np.array(shap_values).squeeze()
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(feature_names, mean_abs_shap_values, color='skyblue')
            plt.xlabel('Mean Absolute SHAP Value')
            plt.title('Feature Importance (SHAP)')
            
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.2f}', 
                        va='center')
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return buf

        plot_buf = plot_shap_bar(shap_values, feature_names=['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows'])
        return plot_buf

    def visualize_model(self) -> BytesIO:
        dummy_input = torch.randn(1, next(iter(self.parameters())).shape[1])
        y = self(dummy_input)
        
        # Generate the graph visualization using torchviz
        dot = make_dot(y, params=dict(self.named_parameters()))
        
        # Render the graph to a temporary file and return it as a BytesIO object
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file_name = tmp_file.name
            dot.render(tmp_file_name, format='png')
        
        with open(f"{tmp_file_name}.png", 'rb') as f:
            img_data = f.read()
        
        os.remove(f"{tmp_file_name}.png")
        
        buf = BytesIO(img_data)
        return buf


    @staticmethod
    def iterate_k_and_evaluate(model: MLP, max_k: int, test_loader: DataLoader, dataset) -> List[dict]:
        results = []
        plots = []
        
        for k in tqdm(range(1, max_k + 1), desc="Evaluating models", unit="model", ascii=True, ncols=100):
            xa_model = XAINT(model, k)
            test_loss = xa_model.test(test_loader)
            plot_buf = xa_model.shap_explain(dataset)
            model_buf = xa_model.visualize_model()
            results.append({'k': k, 'test_loss': test_loss, 'model_summary': model_buf})
            plots.append(plot_buf)
        
        XAINT.save_results(results, plots)
        return results
    

    @staticmethod
    def save_results(results: List[dict], plots: List[BytesIO]):
        n = len(results)
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(16, n * 6))
        
        for i, (result, plot_buf) in enumerate(zip(results, plots)):
            result['model_summary'].seek(0)
            img_summary = Image.open(result['model_summary'])
            axes[i, 0].imshow(img_summary)
            axes[i, 0].axis('off')
            
            plot_buf.seek(0)
            img = Image.open(plot_buf)
            axes[i, 1].imshow(img)
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()






