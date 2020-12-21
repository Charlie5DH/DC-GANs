import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def download_mnist(parameters):
    '''
    Loads MNIST dataset using torch dataloader. Receives
    a dictionary of parameters.
    '''

    transform = transforms.Compose([
        transforms.Resize(parameters['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataloader = DataLoader(
        MNIST('.', download=False, transform=transform),
        batch_size=parameters['batch_size'],
        shuffle=True)

    return dataloader

def download_celeba(parameters):
    '''
    Loads the CELEBA dataset using torch dataloader. Receives
    a dictionary of parameters.
    '''
    transform = transforms.Compose([
        transforms.Resize(parameters['image_size']),
        transforms.CenterCrop('image_size'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = DataLoader(
        CelebA(".", split='train', download=True, transform=transform),
        batch_size=parameters['batch_size'],
        shuffle=True)
    
    return dataloader