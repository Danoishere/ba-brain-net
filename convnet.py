import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torchvision.models as models
import config

batch_size = config.batch_size


# define the NN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()


        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        modules=list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(8192, 2048)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.decoder(enc)
        return x

    def encode(self, x):
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)
        return self.fc1(x)