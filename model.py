import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes,use_resnet=False):
        super(Model,self).__init__()
        self.num_classes = num_classes
        if use_resnet:
            self.model = models.resnet18(weights='DEFAULT') # ResNet18 모델 로드 (Load ResNet18 model)
            # 기존 가중치 고정 (Backbone Freezing)
            for param in self.model.parameters():
                param.requires_grad = False
            
            num_ftrs = self.model.fc.in_features # 마지막 Linear 레이어의 입력 특성 수
            self.model.fc = nn.Linear(num_ftrs, self.num_classes) # 새로운 Linear 레이어로 교체
            # 분류기 부분만 학습 (Train only the classifier part)
            self.head = self.model.fc
        else:
            self.model = models.efficientnet_v2_s(weights='DEFAULT') # MobileNetV3 Large 모델 로드 (Load MobileNetV3 Large model)
            # 기존 가중치 고정 (Backbone Freezing)
            for param in self.model.parameters():
                param.requires_grad = False
            # efficientnet_v2 'classifier' 모듈 내 마지막 Linear 레이어를 가지고 있음
            num_ftrs = self.model.classifier[-1].in_features # 마지막 Linear 레이어의 입력 특성 수
            self.model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes) # 새로운 Linear 레이어로 교체
            # 분류기 부분만 학습 (Train only the classifier part)
            self.head = self.model.classifier[-1]

    #오버라이딩
    def forward(self, x):
        return self.model(x)