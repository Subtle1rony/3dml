from torch import nn

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is a latent vector
            nn.ConvTranspose3d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            # size: (batch_size, 512, 4, 4, 4)

            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            # size: (batch_size, 256, 8, 8, 8)

            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            # size: (batch_size, 128, 16, 16, 16)

            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # size: (batch_size, 64, 32, 32, 32)

            nn.ConvTranspose3d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # size: (batch_size, 1, 64, 64, 64)
        )

    def forward(self, input):
        return self.main(input)

# Define Critic for Wasserstein Loss
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            # input is (batch_size, 1, 64, 64, 64)
            nn.Conv3d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (batch_size, 64, 32, 32, 32)

            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (batch_size, 128, 16, 16, 16)

            nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (batch_size, 256, 8, 8, 8)

            nn.Conv3d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (batch_size, 512, 4, 4, 4)

            nn.Conv3d(512, 1, 4, 1, 0, bias=False),
            # size: (batch_size, 1, 1, 1, 1)
        )

    def forward(self, input):
        return self.main(input)

