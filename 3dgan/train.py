import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Critic
from utils import generate_latent_vectors, save_model, save_sample
from dataset import PointCloudDataset
from tqdm import tqdm

# Directories for saving models and samples
save_dir = './generated_voxels'
model_dir = './models'

# Define hyperparameters
latent_dim = 100
epochs = 10000
batch_size = 32
g_lr = 0.00005
c_lr = 0.00005
n_critic = 5

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the generator and critic
generator = Generator(latent_dim).to(device)
critic = Critic().to(device)

# Initialize the optimizers
g_optim = optim.RMSprop(generator.parameters(), lr=g_lr)
c_optim = optim.RMSprop(critic.parameters(), lr=c_lr)

# Define lambda for gradient penalty
lambda_gp = 10

# Load the dataset
dataset = PointCloudDataset("./voxels/voxels.npy")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = Variable(torch.FloatTensor(*d_interpolates.shape).fill_(1.0), requires_grad=False).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

min_g_loss = float('inf')  # start with infinity

for epoch in range(epochs):
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=150)
    for i, real_data in progress_bar:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # ============================
        # Train the critic
        # ============================
        c_optim.zero_grad()

        # Real data
        real_output = critic(real_data)
        
        # Fake data
        z = generate_latent_vectors(batch_size, latent_dim)
        fake_data = generator(z)
        fake_output = critic(fake_data.detach())

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data, device)

        # Critic loss
        c_loss = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty
        c_loss.backward()
        c_optim.step()

        # ============================
        # Train the generator
        # ============================
        if i % n_critic == 0:
            g_optim.zero_grad()
            z = generate_latent_vectors(batch_size, latent_dim)
            fake_data = generator(z)
            fake_output = critic(fake_data)
            g_loss = -torch.mean(fake_output)
            g_loss.backward()
            g_optim.step()

        progress_bar.set_description(f"Epoch [{epoch + 1}/{epochs}], c_loss: {c_loss.item()}, g_loss: {g_loss.item()}")

    progress_bar.close()

    
    if g_loss < min_g_loss :
        save_sample(generator, epoch, save_dir, device, latent_dim)
    min_g_loss = save_model(generator, g_loss.item(), min_g_loss, model_dir, epoch)
    