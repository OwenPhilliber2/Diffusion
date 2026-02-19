# Imports

from unet import UNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from utils import Diffusion_Utils, sample_timestep, show_tensor_image, prepare_dataset, show_tensor_image

T = 1000
n_samples = 20
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
cd = os.getcwd()

state_dict = torch.load(os.path.join(cd, 'model.pth'), map_location=torch.device(device))
model = UNet()
model.to(device)
model.load_state_dict(state_dict)
model.eval()

diffusion_utils = Diffusion_Utils(num_timesteps=T, device=device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = prepare_dataset([3, 4, 5, 7], transform = transform, download = False) # cat, deer, dog, horse
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Get a sample image from the dataset and display it
dataiter = iter(dataloader)
images, labels = next(dataiter)
guide_img = images[0]

# Sampling images and creating a plot of the denoising process

@torch.no_grad()
def sample_guided(guide_img, plot_steps = True, plot = True):
    guide_img = guide_img.to(device)
    img_size = 32
    img = .95 * torch.randn((1, 3, img_size, img_size), device=device)
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
        if 200 <= i <= 400:
            # Adding the sample image at certain intervals to the diffusion process
            img = img + 0.005 * guide_img

    if not plot_steps:
        show_tensor_image(img_vis.detach().cpu())
        
    if plot:
        plt.show()   

    return plt

# Get a sample image from the dataset and display it
for i in range(n_samples):
    guide_img = images[i]
    plt.subplot(2,5,1)
    sample_guided(guide_img, plot_steps=False, plot=False)
    plt.subplot(2,5,2)
    sample_guided(guide_img, plot_steps=False, plot=False)
    plt.subplot(2,5,3)
    sample_guided(guide_img, plot_steps=False, plot=False)
    plt.subplot(2,5,4)
    sample_guided(guide_img, plot_steps=False, plot=False)
    plt.subplot(2,5,5)
    sample_guided(guide_img, plot_steps=False, plot=False)
    plt.subplot(2,5,6)
    show_tensor_image(guide_img)
    plt.savefig(os.path.join('imgs', f'guided_sample_{i+1}.png'))
    # plt.show()
