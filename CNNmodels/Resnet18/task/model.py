from torch import nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=47):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)
