import torch
import torch.nn as nn
import torchvision.models as models


class VisualPerceptionNet(nn.Module):
    def __init__(self, input_channels=3, visual_feature_size=24):
        super(VisualPerceptionNet, self).__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=True)
        self.squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for param in self.squeezenet.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, visual_feature_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.squeezenet.features(x)  # SqueezeNet backbone is frozen
            x = self.pool(x)  # Adaptive pooling to reduce spatial dimensions to 1x1
        x = torch.flatten(x, 1)  # Flatten to shape [batch_size, 512]
        
        # FC layer
        x = self.fc(x)
        return x