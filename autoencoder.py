import torch.nn as nn
import torch.nn.functional as F

from convlstm import ConvLSTM

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, torchDevice):
        super(ConvAutoencoder, self).__init__()

        self.pre = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.clstm = ConvLSTM(
                torchDevice,
                input_size=(32, 32),
                 input_dim=128,
                 hidden_dim=[256, 512],
                 kernel_size=(3, 3),
                 num_layers=2,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False)

        self.static_replacement = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.post = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        self.batchnorm = nn.BatchNorm2d(512)

    def reset_hidden_state(self):
        self.hidden = None

    def forward(self, x):
        enc = self.post(x)
        x = self.decoder(enc)
        return x

    def encode(self, x):
        x = self.pre(x)
        x = x.unsqueeze(0)
        x, self.hidden = self.clstm(x, self.hidden)
        x = x.squeeze(0)
        x = self.batchnorm(x)
        return x

    def encode_static(self, x):
        x = self.pre(x)
        x = self.static_replacement(x)
        x = self.batchnorm(x)
        return x

    def post_clstm(self, x):
        return self.post(x)