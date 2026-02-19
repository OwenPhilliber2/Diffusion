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

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

class Diffusion_Utils:
    def __init__(self, num_timesteps=1000, cos_schedule = False, beta_start=1e-4, beta_end=2e-2, device=None):
        self.device = device


        if cos_schedule:
            raise NotImplementedError("Cosine schedule not implemented yet")
        else:
            self.betas = torch.linspace(start=beta_start, end=beta_end, steps=num_timesteps, device=self.device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)


        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """ Diffuse image x_0 to timestep t"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return (
            sqrt_alphas_cumprod_t * x_0
            + sqrt_one_minus_alphas_cumprod_t * noise
        ), noise
    
    def _get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_loss(self, model, x_0, t):
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        x_noisy, noise = self.q_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        
        return F.mse_loss(noise, noise_pred)
    
# Sampling images and creating a plot of the denoising process

@torch.no_grad()
def sample_timestep(x, t, diffusion_utils=Diffusion_Utils(), model = UNet(), device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = diffusion_utils._get_index_from_list(diffusion_utils.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = diffusion_utils._get_index_from_list(
        diffusion_utils.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    
    
    
    sqrt_recip_alphas_t = diffusion_utils._get_index_from_list(diffusion_utils.sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    posterior_variance_t = diffusion_utils._get_index_from_list(diffusion_utils.posterior_variance, t, x.shape)
    
    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(plot_steps = True, device = "cuda" if torch.cuda.is_available() else "cpu", T = 1000):
    
    img_size = 32
    img = torch.randn((1, 3, img_size, img_size), device=device)
    if plot_steps:
        plt.figure(figsize=(15,2))
    else:
        plt.figure(figsize=(5,5))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img_vis = torch.clamp(img, -1.0, 1.0)
        if plot_steps:
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                show_tensor_image(img_vis.detach().cpu())
    
    if not plot_steps:
        show_tensor_image(img_vis.detach().cpu())
    
    plt.show()

def prepare_dataset(label_list, download=True, transform=None):
    '''
    Downloads and prepares the data

    label_ind - index of the desired label to make the diffusion model on
    The list is, ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    '''
    # Downloading and preparing the CIFAR-10 dataset
    train_data = torchvision.datasets.CIFAR10('./data', train=True, download=download, transform=transform)
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=download, transform=transform)

    train_label_indices = [i for i, label in enumerate(train_data.targets) if label in label_list]
    test_label_indices = [i for i, label in enumerate(test_data.targets) if label in label_list]

    train = Subset(train_data, train_label_indices)
    test = Subset(test_data, test_label_indices)

    return torch.utils.data.ConcatDataset([train, test])
   