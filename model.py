import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

return_nodes = {
    "layer4": "layer4",
}


class ResNetModel(nn.Module):
    "ResNet model followed by two FFNs"

    def __init__(self):
        super().__init__()
        model = resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(
            model, return_nodes=return_nodes
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(65536, 16 * 8)
        self.linear2 = nn.Linear(16 * 8, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.feature_extractor(x)["layer4"]))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
