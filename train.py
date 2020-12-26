import sys
import os
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm

from utils import show_tensor_images, download_mnist, download_celeba, plot_losses
from DCGAN import Generator, Discriminator, weights_init

seed = 42
torch.manual_seed(seed)

# Parameters to define the model.
params = {
    "batch_size"   : 128,      # Batch size during training.
    'image_size'   : 64,       # Spatial size of training images.
    'img_chan'     : 1,        # Number of channles of images. 3 for RGB.
    'z_dim'        : 64,       # the dimension of the noise vector, a scalar
    'hidden_dim'   : 64,       # Size of feature maps in the generator.
    'kernel_size'  : 4,
    'stride'       : 2,
    'd_hidden_dim' : 64,       # Size of features maps in the discriminator.
    'n_epochs'     : 10,       # Number of training epochs.
    'lr'           : 0.0002,   # Learning rate for optimizers
    'beta_1'       : 0.5,      # Beta1 hyperparam for Adam optimizer
    'beta_2'       : 0.999,    # Beta2 hyperparam for Adam optimizer
    'save_epoch'   : 2,
    'output_chn'   : 1,        # Number of channels of the output image 
    'device'       : 'cpu',
    'download'     : 'True',
    'data_dir'     : 'dataset/'
    }

def get_noise(n_samples, params):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
    '''
    # pylint: disable=E1101
    return torch.randn(n_samples, params['z_dim'], device=params['device'])
    # pylint: enable=E1101
    # solving Error: module 'torch' has no 'randn' member

def get_train_data(params):
    if params['img_chan'] == 1:
        dataloader = download_mnist(params)
    else:
        dataloader = download_celeba(params)
    return dataloader

def create_generator(params):
    '''
    Define generator and optmizer.
    '''
    gen = Generator(params).to(params['device'])
    gen_opt = torch.optim.Adam(gen.parameters(), lr=params['lr'], betas=(params['beta_1'],params['beta_2']))
    gen = gen.apply(weights_init)
    return gen, gen_opt

def create_discriminator(params):

    disc = Discriminator(params).to(params['device']) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=params['lr'], betas=(params['beta_1'],params['beta_2']))
    disc = disc.apply(weights_init)
    return disc, disc_opt

def train(params):
    '''
    Train the GAN network.
    '''
    criterion = nn.BCEWithLogitsLoss()

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = get_train_data(params)
    display_step = 500
    g_loss = []
    d_loss = []

    gen, gen_opt = create_generator(params)
    disc, disc_opt = create_discriminator(params)

    for epoch in range(params['n_epochs']):
    # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(params['device'])

            ## Update discriminator ##
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, params)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            # pylint: disable=E1101
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            # pylint: enable=E1101

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()

            ## Update generator ##
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, params)
            fake_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fake_2)
            # pylint: disable=E1101
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # pylint: enable=E1101
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            g_loss.append(gen_loss.item())
            d_loss.append(disc_loss.item())

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            
            cur_step += 1
    
    plot_losses(g_loss, d_loss)

    # Save the trained model.
    torch.save({
    'generator' : gen.state_dict(),
    'discriminator' : disc.state_dict(),
    'optimizerG' : gen_opt.state_dict(),
    'optimizerD' : disc_opt.state_dict(),
    'params' : params
    }, 'model/model_final.pth')

def main():

    # Use GPU is available else use CPU.
    # pylint: disable=E1101
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    # pylint: enable=E1101
    print(device, " will be used.\n")
    params['device'] = device

    parser = argparse.ArgumentParser(description='Train DCGAN on MNIST or CelebA datasets')
    parser.add_argument('-batch_size', metavar='bz', type=int, help='Batch size during training.', default=128)
    parser.add_argument('-image_size', type=int, help='Spatial size of training images.', default=64)
    parser.add_argument('-z_dim', type=int, help='The dimension of the noise vector, a scalar', default=64)
    parser.add_argument('-hidden_dim', type=int, help='Size of feature maps in the generator.', default=64)
    parser.add_argument('-d_hidden_dim', type=int, help='Size of features maps in the discriminator.', default=64)
    parser.add_argument('-n_epochs', type=int, help='Number of training epochs.', default=10)
    parser.add_argument('-data_dir', help='Directory', default='dataset/')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-mni', '--train_mnist', help='Train MNSIT dataset', action='store_false')
    group.add_argument('-cel', '--train_celeb', help='Train CelebA dataset', action='store_false')
    
    args = parser.parse_args()

    params['batch_size'] = args.batch_size
    params['image_size'] = args.image_size
    params['z_dim'] = args.z_dim
    params['hidden_dim'] = args.hidden_dim
    params['d_hidden_dim'] = args.d_hidden_dim
    params['n_epochs'] = args.n_epochs
    params['data_dir'] = args.data_dir

    if args.train_mnist:
        params['img_chan'] = 1
        params['output_chn'] = 1
    elif args.train_celeb:
        params['img_chan'] = 3
        params['output_chn'] = 3
    else:
        pass

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)
    
    train(params)

if __name__ == "__main__":
    main()
