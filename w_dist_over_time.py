# Imports

from unet import UNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
from utils import Diffusion_Utils, sample_timestep, show_tensor_image

T = 1000
n_samples = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
cd = os.getcwd()

state_dict = torch.load(os.path.join(cd, 'model.pth'), map_location=torch.device(device))
model = UNet()
model.to(device)
model.load_state_dict(state_dict)
model.eval()

diffusion_utils = Diffusion_Utils(num_timesteps=T, device=device)

# Sampling images and creating a plot of the denoising process

@torch.no_grad()
def sample_w_dist_over_time(plot_steps = True, plot_distances = True, plot_img = True):
    
    img_size = 32
    img = torch.randn((1, 3, img_size, img_size), device=device)
    img0 = img.clone()
    w_distances = []
    if plot_steps:
        plt.figure(figsize=(15,2))
        plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model = model)
        img_vis = torch.clamp(img, -1.0, 1.0)
        if plot_steps:
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                show_tensor_image(img_vis.detach().cpu())
        if i % 50 == 0:
            # Calculate the Wasserstein distance between the current image and the original image
            img_flat = img_vis.view(-1)
            img0_flat = img0.view(-1)
            w_distances.append(F.pairwise_distance(img_flat.unsqueeze(0), img0_flat.unsqueeze(0)).item())

    if plot_steps:
        plt.show()

    if plot_img:
        plt.figure(figsize=(5,5))
        plt.axis('off')
        show_tensor_image(img_vis.detach().cpu())
        plt.show()

    if plot_distances:
        # Plot the Wasserstein distance over time
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(0, T, 50), w_distances)
        plt.xlabel('Timestep')
        plt.ylabel('Wasserstein Distance')
        plt.show()

    return w_distances
    
    
distance_list = []

for i in range(n_samples):
    w_distances = sample_w_dist_over_time(plot_steps=False, plot_distances=False, plot_img = False)   
    distance_list.append(w_distances)

# Plot the Wasserstein distance over time for all samples
plt.figure(figsize=(10,5))
for i in range(n_samples):
    plt.plot(np.arange(0, T, 50), distance_list[i], label=f'Sample {i+1}')
plt.xlabel('Timestep')
plt.ylabel('Wasserstein Distance')
plt.title('Wasserstein Distance over Time for Multiple Samples')
plt.legend()
plt.show()