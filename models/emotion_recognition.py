import torch
import torch.nn as nn

from torch.optim.adam import Adam
from sklearn.metrics import accuracy_score

class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 emotions
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, epochs=10):
        """Train the model using the provided training DataLoader."""
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

        print('Finished Training')

    def evaluate_model(self, test_loader):
        """Evaluate the model using the provided test DataLoader."""
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Accuracy: {accuracy:.4f}')



# model = EmotionRecognitionModel()

# model.train_model(train_loader, epochs=10)
# model.evaluate_model(test_loader)
