import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50(
    num_classes,
    pretrained=True,
    freeze_backbone=False
):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model
