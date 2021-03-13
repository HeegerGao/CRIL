import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
import shutil
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from utils import CustomDataSetForFirstFrame
from models import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval betwen image samples")
parser.add_argument("--ngf", type=int, default=32, help="channels of generator")
parser.add_argument("--ndf", type=int, default=32, help="channels of discriminitor")
arg = parser.parse_args()

img_shape = (arg.channels, arg.img_size, arg.img_size)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    d_interpolates = torch.reshape(d_interpolates, (d_interpolates.shape[0], d_interpolates.shape[1])).cuda()
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(
    data_path: List,
    env_name: List,
    load_old_models,
    old_g_path:str=None,
    old_d_path:str=None,
):
    task_name = ''
    for name in env_name:
        task_name += name
        task_name += '_'

    root_path = 'trained_generators/' + task_name
    os.makedirs(root_path, exist_ok=True)
    log_dir = 'trained_generators/log/' + task_name
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if load_old_models == True:
        generator = torch.load(old_g_path)
        discriminator = torch.load(old_d_path)


    if cuda:
        generator.cuda()
        discriminator.cuda()

    dataloader = torch.utils.data.DataLoader(
        dataset=CustomDataSetForFirstFrame(
            paths=data_path,
        ),
        batch_size=arg.batch_size,
        shuffle=False,
    )


    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=arg.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=arg.lr)

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(arg.n_epochs):
        for i, imgs in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], arg.latent_dim, 1, 1).cuda()

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % arg.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()


                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, arg.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                writer.add_scalar('D loss', d_loss.item(), batches_done)
                writer.add_scalar('G loss', g_loss.item(), batches_done)

                if batches_done % arg.sample_interval == 0:
                    torch.save(generator, root_path+'/G'+str(batches_done)+'.pth')
                    torch.save(discriminator, root_path+'/D'+str(batches_done)+'.pth')
                    save_image(fake_imgs.data[:16], root_path+'/%d.png' % batches_done, nrow=4, normalize=True)
                batches_done += arg.n_critic