import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(42)

def weights_init(m):
    '''
    Initialize the weights to the normal distribution
    with mean 0 and standard deviation 0.02
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

###----------------------------------Generator-------------------------------###
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

        self.gen = nn.Sequential(
            self.gen_block(self.z_dim, self.hidden_dim * 4),
            self.gen_block(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=4, stride=1),
            self.gen_block(self.hidden_dim * 2, self.hidden_dim),
            self.gen_block(self.hidden_dim, self.img_chan, kernel_size=4, final_layer=True),
        )

    def gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Make a generator block base on Transposed convolutions
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
        '''

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()                
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
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

###----------------------------------Discriminator-------------------------------##
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
            self.disc_block(self.img_chan, self.hidden_dim),
            self.disc_block(self.hidden_dim, self.hidden_dim * 2),
            self.disc_block(self.hidden_dim * 2, self.output_channels, final_layer=True),
        )

    def disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)          
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
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


