import torch
import torch.nn as nn
from torchvision import models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=14, is_trained=True):
        """
        DenseNet121 model for Chest X-ray classification.
        
        Args:
            num_classes (int): Number of target classes.
            is_trained (bool): Whether to use pretrained ImageNet weights.
        """
        super(DenseNet121, self).__init__()
        
        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.DEFAULT if is_trained else None
        self.densenet121 = models.densenet121(weights=weights)
        
        # Get input features of the last layer
        num_features = self.densenet121.classifier.in_features
        
        # Replace classifier
        # Using a simple linear layer as is standard for CheXNet/CheXpert implementations
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
            # Sigmoid is omitted here to use BCEWithLogitsLoss for numerical stability via training
        )

    def forward(self, x):
        return self.densenet121(x)
