import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, num_classes, model_name='tf_efficientnetv2_s'):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        self.head = self.model.get_classifier()
        # classification layer의 파라미터 동결 해제
        for param in self.head.parameters():
            param.requires_grad = True
    # 오버라이딩
    def forward(self, x):
        return self.model(x)