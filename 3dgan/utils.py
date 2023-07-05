import torch
import numpy as np
import os

def generate_latent_vectors(batch_size, latent_dim):
    # Generates a batch of latent vectors
    return torch.randn((batch_size, latent_dim, 1, 1, 1), device='cuda')

def save_model(generator, g_loss, min_g_loss, model_dir, epoch):
    if g_loss < min_g_loss and epoch > 10:
        print(f"New low generator loss, saving model...")
        min_g_loss = g_loss
        torch.save(generator, os.path.join(model_dir, 'generator.pt'))
    return min_g_loss

def save_sample(generator, epoch, save_dir, device, latent_dim):
    with torch.no_grad():
        z = generate_latent_vectors(1, latent_dim).to(device)  # generate one sample
        sample = generator(z)
    np.save(os.path.join(save_dir, f'generated_sample_{epoch+1}.npy'), sample.cpu().numpy())

if __name__ == '__main__':
    generator = torch.load('./models/generator.pt')
    with torch.no_grad(): 
        z = generate_latent_vectors(1, 100).to('cuda')  # generate one sample
        sample = generator(z)
    np.save(os.path.join('./generated_voxels', f'generated_sample_{999}.npy'), sample.cpu().numpy())