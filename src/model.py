import torch
import torch.nn as nn
from torchvision import models


class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        # load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # freeze early layers — keep ImageNet features
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # replace final classifier with 7-class output
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = EmotionNet(num_classes=7).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params    : {total:,}')
    print(f'Trainable params: {trainable:,}')

    dummy  = torch.randn(4, 3, 48, 48).to(device)
    output = model(dummy)
    print(f'Input shape : {dummy.shape}')
    print(f'Output shape: {output.shape}')