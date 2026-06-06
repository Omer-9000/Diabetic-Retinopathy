import torch
import torch.nn as nn

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,

    efficientnet_b3,
    EfficientNet_B3_Weights,

    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,

    convnext_tiny,
    ConvNeXt_Tiny_Weights,

    swin_t,
    Swin_T_Weights
)


# ==========================================================
# CUSTOM DIABETIC RETINOPATHY CNN
# ==========================================================

class CustomDRCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================================
# MODEL FACTORY
# ==========================================================

def get_model(
    model_name,
    num_classes=4,
    pretrained=True
):

    # ------------------------------------------------------
    # CUSTOM CNN
    # ------------------------------------------------------

    if model_name == "custom_cnn":

        return CustomDRCNN(
            num_classes=num_classes
        )

    # ------------------------------------------------------
    # DENSENET121
    # ------------------------------------------------------

    elif model_name == "densenet121":

        weights = (
            DenseNet121_Weights.DEFAULT
            if pretrained else None
        )

        model = densenet121(
            weights=weights
        )

        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes
        )

        return model

    # ------------------------------------------------------
    # EFFICIENTNET B3
    # ------------------------------------------------------

    elif model_name == "efficientnet_b3":

        weights = (
            EfficientNet_B3_Weights.DEFAULT
            if pretrained else None
        )

        model = efficientnet_b3(
            weights=weights
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

        return model

    # ------------------------------------------------------
    # EFFICIENTNET V2 S
    # ------------------------------------------------------

    elif model_name == "efficientnet_v2_s":

        weights = (
            EfficientNet_V2_S_Weights.DEFAULT
            if pretrained else None
        )

        model = efficientnet_v2_s(
            weights=weights
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

        return model

    # ------------------------------------------------------
    # CONVNEXT TINY
    # ------------------------------------------------------

    elif model_name == "convnext_tiny":

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )

        model = convnext_tiny(
            weights=weights
        )

        in_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(
            in_features,
            num_classes
        )

        return model

    # ------------------------------------------------------
    # SWIN TRANSFORMER TINY
    # ------------------------------------------------------

    elif model_name == "swin_t":

        weights = (
            Swin_T_Weights.DEFAULT
            if pretrained else None
        )

        model = swin_t(
            weights=weights
        )

        in_features = model.head.in_features

        model.head = nn.Linear(
            in_features,
            num_classes
        )

        return model

    # ------------------------------------------------------
    # UNKNOWN MODEL
    # ------------------------------------------------------

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model_names = [
        "custom_cnn",
        "densenet121",
        "efficientnet_b3",
        "efficientnet_v2_s",
        "convnext_tiny",
        "swin_t"
    ]

    for name in model_names:

        model = get_model(
            name,
            num_classes=4,
            pretrained=True
        )

        params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"{name:<20} "
            f"{params/1e6:.2f}M parameters"
        )