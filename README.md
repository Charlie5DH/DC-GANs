# DC-GANs
Deep Convolutional GANs implementation repo in PyTorch.



## Requirements

- PyTorch 1.7
- matplotlib
- Python3

## Set up the Python environment

Run `conda env create` to create an environment called `DCGANS`, as defined in `environment.yml`. This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries. (`conda env create -f environment.yml`). We will install Python libraries using `pip-sync`, however, which will let us do three nice things:

Or you can run

```
conda env create --prefix ./env --file environment.yml
```

To create the environment as sub-directory

So, after running `conda env create`, activate the new environment and install the requirements:

```sh
conda activate DCGANS or conda activate ./env
pip install -r requirements.txt
```
If `pip install -r requirements` fails when installing torchvision install it mannually.
`pip install torchvision===0.8.2 -f https://download.pytorch.org/whl/torch_stable.html`

If you add, remove, or need to update versions of some requirements, edit the `.in` files, then run

```
pip-compile requirements.in && pip-compile requirements-dev.in
```

## Use

For training run the training script `train.py` and pass the argumenrs.

### Default arguments

```python
parameters = {
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
```

## References

- MNIST Database: http://yann.lecun.com/exdb/mnist/
- CelebFaces Attributes Dataset (CelebA): http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): https://arxiv.org/abs/1701.07875
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016): https://arxiv.org/abs/1511.06434
- Deconvolution and Checkerboard Artifacts (Odena et al., 2016): http://doi.org/10.23915/distill.00003
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
