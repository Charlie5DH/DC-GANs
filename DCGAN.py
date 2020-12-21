import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(42)

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, parameters):
        super(Generator, self).__init__()
        self.z_dim = parameters['zdim']
        self.hidden_dim = parameters['hidden_dim']
        self.kernel_size = parameters['kernel_size']
        self.stride = parameters['stride']
        self.img_chan = parameters['img_chan']
        
        # Build the neural network
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim*4, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU()
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*8, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU()
            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU()
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU()
            nn.ConvTranspose2d(hidden_dim*2, img_chan, kernel_size, stride),
            nn.Tanh()
        )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

    def get_noise(self, n_samples, z_dim, device='cpu'):
        '''
        Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
        creates a tensor of that shape filled with random numbers from the normal distribution.
        Parameters:
            n_samples: the number of samples to generate, a scalar
            z_dim: the dimension of the noise vector, a scalar
            device: the device type
        '''
        return torch.randn(n_samples, z_dim, device=device)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    im_chan: the number of channels in the images, fitted for the dataset used, a scalar
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, parameters, output_channels):
        super(Discriminator, self).__init__()

        self.hidden_dim = parameters['hidden_dim']
        self.kernel_size = parameters['kernel_size']
        self.stride = parameters['stride']
        self.img_chan = parameters['img_chan']
        self.output_channels = output_channels

        self.disc = nn.Sequential(
            nn.Conv2d(img_chan, hidden_dim, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(img_chan, hidden_dim*2, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_dim*2, output_channels, kernel_size, stride)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


