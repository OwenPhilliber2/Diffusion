## Diffusion project
Building a diffusion generative model on the CIFAR10 dataset. The model is based on the papers DDPM, https://arxiv.org/pdf/2006.11239, and improved DDPM, https://arxiv.org/abs/2102.09672.

## Overview:

In order to train a diffusion model, run diffusion.ipynb. This trains a UNet model from unet.py on a subset of the CIFAR10 dataset with only the images labeled as cat, deer, dog, and horse.

A pretrained model as well as sample images can be found at: https://drive.google.com/drive/folders/1fVJoWfnMPWayZ_YyurSIJOxW-O2v_zD9?usp=drive_link. In order to use the pretrained model, download the model.pth folder and place it in the Diffusion folder. 

w_dist_over_time.py - Samples 5 images from the diffusion model and calculates the Wasserstein distance from the initial random gaussian image to the diffused images every 50 steps. It then plots the distance over time for each of the samples. Using this we can see that the distances for the first 200 steps are similar for each sample. My hypothesis is that adding extra noise to steps roughly 200-400, we can influence the final image.

guided_sample.py - WORK IN PROGRESS - Attempts to guide the diffusion model by adding a 0.005 times the desired image at each time step between 200 and 400. The images produced by this method often times result in similar distinctive features to the desired image, but they are not well created images. More work will be done to improve this process.

