import torch.nn as nn
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

class Model(nn.Module):
    def __init__(self, num_classes, model_name='tf_efficientnetv2_s'):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.requires_grad_(False)
        self._classifier = self.model.get_classifier()
        self._classifier.requires_grad_(True)
        self.norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def forward(self, x):
        return self.model(self.norm(x))
        
    @property
    def classifier(self):
        return self._classifier
