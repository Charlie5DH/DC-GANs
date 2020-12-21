import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(42)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

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
        self.z_dim = parameters['z_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.kernel_size = parameters['kernel_size']
        self.stride = parameters['stride']
        self.img_chan = parameters['img_chan']
        
        # Build the neural network
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.hidden_dim*4, kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.hidden_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim*4, self.hidden_dim*8, self.kernel_size, self.stride),
            nn.BatchNorm2d(self.hidden_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim*8, self.hidden_dim*4, self.kernel_size, self.stride),
            nn.BatchNorm2d(self.hidden_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim*4, self.hidden_dim*2, self.kernel_size, self.stride),
            nn.BatchNorm2d(self.hidden_dim*2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim*2, self.img_chan, self.kernel_size, self.stride),
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

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    im_chan: the number of channels in the images, fitted for the dataset used, a scalar
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, parameters):
        super(Discriminator, self).__init__()

        self.hidden_dim = parameters['d_hidden_dim']
        self.kernel_size = parameters['kernel_size']
        self.stride = parameters['stride']
        self.img_chan = parameters['img_chan']
        self.output_channels = parameters['output_chn']

        self.disc = nn.Sequential(
            nn.Conv2d(self.img_chan, self.hidden_dim, self.kernel_size, self.stride),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, self.stride),
            nn.BatchNorm2d(self.hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.hidden_dim*2, self.output_channels, self.kernel_size, self.stride)
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


