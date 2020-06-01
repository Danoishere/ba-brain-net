import torch.nn as nn
import torch
from convlstm import ConvLSTMCell

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, torchDevice):
        super(ConvAutoencoder, self).__init__()
        self.torchDevice = torchDevice
        self.pre_encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(4, 128, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=5, stride=2, padding=1),
        )

        self.clstm = ConvLSTMCell(torchDevice, 2048, 2048).to(torchDevice)

        self.post_encoder = nn.Sequential( # like the Composition layer you built
            nn.ReLU(),
            nn.Conv2d(2048, 128, kernel_size=3)
        )

    def reset_hidden_state(self):
        self.state = None

    def forward(self, x):
        enc = self.pre_encoder(x)
        self.state = self.clstm(enc, self.state)
        enc = self.post_encoder(self.state[0])
        return enc