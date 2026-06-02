import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=5, pretrained=True):
    """
    Factory function to return the requested model with a modified classification head.
    """
    weights = 'DEFAULT' if pretrained else None
    
    if model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    return model
