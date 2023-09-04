import DiscriminatorTrainingImg from '../assets/coursera/build_basic_gans/discriminator_training.jpg'
import GeneratorTrainingImg from '../assets/coursera/build_basic_gans/generator_training.jpg'

export const markdownContent = `
# GAN Implementation in PyTorch

Generative Adversarial Networks (GANs) are widely used to generate realistic images. They use two neural networks which competes against each other. One is Generator and other is Discriminator.



## Discriminator
The Discriminator is a neural network whose task is to classify the given image as real or fake. The goal of the discriminator is to model the probability of each class given a set of input features.
i.e. -
$$
  p(y|x)
$$

where $y$ is the class and $x$ is the input features.

![Discriminator Training](${DiscriminatorTrainingImg})





## Generator
The Generator is a neural network whose task is to generate fake images. It takes a random noise as input and generates an image.
The goal of the generator is to fool the discriminator by generating fake images that are indistinguishable from real images.
Once the generator is trained, you freeze its parameters and use it to generate images by passing random noise as input. This is known as Sampling.

Thus Generator gives - 
$$
  p(x|y)
$$
i.e. - probability of generating an image $x$ given a class $y$.

![Generator Training](${GeneratorTrainingImg})





## Implementation
The following code snippet shows how to implement a GAN in Python using PyTorch.
`


export const codeString = `# import libraries
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

# function to display images
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
  '''
  Function for visualizing images: Given a tensor of images, number of images, and size per image, plots and prints the image in a uniform grid.
  '''
  image_unflat = image_tensor.detach().cpu().view(-1, *size)
  image_grid = make_grid(image_unflat[:num_images], nrow=5)
  plt.imshow(image_grid.permute(1, 2, 0).squeeze())
  plt.show()

# generator block
def get_generator_block(input_dim, output_dim):
  '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation
          followed by a batch normalization and then a relu activation
  '''
  return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.BatchNorm1d(output_dim),
      nn.ReLU(inplace=True)
  )

# generator class
class Generator(nn.Module):
  '''
  Generator Class
  Values:
      z_dim: the dimension of the noise vector, a scalar
      im_dim: the dimension of the images, fitted for the dataset used, a scalar
        (MNIST images are 28 x 28 = 784 so that is your default)
      hidden_dim: the inner dimension, a scalar
  '''
  def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
    super(Generator, self).__init__()

    # Build the neural network
    self.gen = nn.Sequential(
        get_generator_block(z_dim, hidden_dim),
        get_generator_block(hidden_dim, hidden_dim * 2),
        get_generator_block(hidden_dim * 2, hidden_dim * 4),
        get_generator_block(hidden_dim * 4, hidden_dim * 8),
        # the last layer is different
        nn.Linear(hidden_dim * 8, im_dim),
        nn.Sigmoid()
    )

  # Forward pass method
  def forward(self, noise):
    '''
    Function for completing a forward pass of the generator: Given a noise tensor,
    returns generated images.
    Parameters:
        noise: a noise tensor with dimensions (n_samples, z_dim)
    '''
    return self.gen(noise)

# function to get noise
def get_noise(n_samples, z_dim, device='cuda'):
  '''
  Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
  creates a tensor of that shape filled with random numbers from the normal distribution.
  Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
  '''
  return torch.randn(n_samples, z_dim, device=device)


# discriminator block
def get_discriminator_block(input_dim, output_dim):
  '''
  Discriminator Block
  Function for returning a neural network of the discriminator given input and output dimensions.
  Parameters:
      input_dim: the dimension of the input vector, a scalar
      output_dim: the dimension of the output vector, a scalar
  Returns:
      a discriminator neural network layer, with a linear transformation
        followed by an nn.LeakyReLU activation with negative slope of 0.2
        (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
  '''
  return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.LeakyReLU(negative_slope=0.2)
  )

# discriminator class
class Discriminator(nn.Module):
  '''
  Discriminator Class
  Values:
      im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28x28 = 784 so that is your default)
      hidden_dim: the inner dimension, a scalar
  '''
  def __init__(self, im_dim=784, hidden_dim=128):
    super(Discriminator, self).__init__()

    # Build the neural network
    self.disc = nn.Sequential(
      get_discriminator_block(im_dim, hidden_dim * 4),
      get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
      get_discriminator_block(hidden_dim * 2, hidden_dim),
      nn.Linear(hidden_dim, 1)
    )


  # Forward pass method
  def forward(self, image):
    '''
    Function for completing a forward pass of the discriminator: Given an image tensor,
    returns a 1-dimension tensor representing fake/real.
    Parameters:
        image: a flattened image tensor with dimension (im_dim)
    '''
    return self.disc(image)


# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

device = 'cuda'


# Initialize generator and discriminator
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# function to calculate discriminator loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
  '''
  Return the loss of the discriminator given inputs.
  Parameters:
      gen: the generator model, which returns an image given z-dimensional noise
      disc: the discriminator model, which returns a single-dimensional prediction of real/fake
      criterion: the loss function, which should be used to compare
              the discriminator's predictions to the ground truth reality of the images
              (e.g. fake = 0, real = 1)
      real: a batch of real images
      num_images: the number of images the generator should produce,
              which is also the length of the real images
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
  Returns:
      disc_loss: a torch scalar loss value for the current batch
  '''
  noise_vectors = get_noise(num_images, z_dim, device=device)
  fake_images = gen(noise_vectors)

  disc_preds_for_fakes = disc(fake_images.detach())
  true_labels_for_fakes = torch.zeros_like(disc_preds_for_fakes)
  disc_loss_for_fakes = criterion(disc_preds_for_fakes, true_labels_for_fakes)

  disc_preds_for_reals = disc(real)
  true_labels_for_reals = torch.ones_like(disc_preds_for_reals)
  disc_loss_for_reals = criterion(disc_preds_for_reals, true_labels_for_reals)

  disc_loss = (disc_loss_for_fakes + disc_loss_for_reals) / 2

  return disc_loss

# function to calculate generator loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
  '''
  Return the loss of the generator given inputs.
  Parameters:
      gen: the generator model, which returns an image given z-dimensional noise
      disc: the discriminator model, which returns a single-dimensional prediction of real/fake
      criterion: the loss function, which should be used to compare
              the discriminator's predictions to the ground truth reality of the images
              (e.g. fake = 0, real = 1)
      num_images: the number of images the generator should produce,
              which is also the length of the real images
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
  Returns:
      gen_loss: a torch scalar loss value for the current batch
  '''
  noise_vectors = get_noise(num_images, z_dim, device=device)
  fake_images = gen(noise_vectors)

  disc_preds_for_fakes = disc(fake_images)
  gen_loss = criterion(disc_preds_for_fakes, torch.ones_like(disc_preds_for_fakes))

  return gen_loss


# Actual training
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False
error = False

for epoch in range(n_epochs):

  # Dataloader returns the batches
  for real, _ in tqdm(dataloader):
    cur_batch_size = len(real)

    # Flatten the batch of real images from the dataset
    real = real.view(cur_batch_size, -1).to(device)

    ### Update discriminator ###
    disc_opt.zero_grad()
    disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    ### Update generator ###
    gen_opt.zero_grad()
    gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
    gen_loss.backward(retain_graph=True)
    gen_opt.step()


    # keep track of the average disc loss
    mean_discriminator_loss += disc_loss.item() / display_step

    # Keep track of the average generator loss
    mean_generator_loss += gen_loss.item() / display_step

    ### Visualization code ###
    if cur_step % display_step == 0 and cur_step > 0:
      print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
      fake_noise = get_noise(cur_batch_size, z_dim, device=device)
      fake = gen(fake_noise)
      show_tensor_images(fake)
      show_tensor_images(real)
      mean_generator_loss = 0
      mean_discriminator_loss = 0
    cur_step += 1
`


