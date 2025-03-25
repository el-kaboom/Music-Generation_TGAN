# Discriminator model 
import torch
import torch.nn as nn

class MusicDiscriminator(nn.Module):
    def __init__(self, input_dim=512):
        super(MusicDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
