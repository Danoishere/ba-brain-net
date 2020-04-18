import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(4, 128, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        enc = self.encoder(x)
        x = self.decoder(enc)
        return x

    def encode(self, x):
        return self.encoder(x)