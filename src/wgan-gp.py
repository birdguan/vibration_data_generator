import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time
import matplotlib.pyplot as plt
import os
from torchvision import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.parameters import *

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main_module = nn.Sequential(
            # 噪声 -> (4, 4, ngf*8)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (4, 4, ngf*8) -> (8, 8, ngf*4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # (8, 8, ngf*4) -> (16, 16, ngf*2)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (16, 16, ngf*2) -> (32, 32, ngf*2)
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (32, 32, ngf*2) -> (64, 64, ngf)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (64, 64, ngf) -> (128, 128, ngf)
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (128, 128, ngf) -> (256, 256, nc)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

	def forward(self, x):
		return self.main_module(x)


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main_module = nn.Sequential(
            # (256, 256, nc) -> (128, 128, ndf)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 128, ndf) -> (64, 64, ndf)
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 64, ndf) -> (32, 32, ndf*2)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (32, 32, ndf*2) -> (16, 16, ndf*2)
            nn.Conv2d(ndf*2, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (16, 16, ndf*2) -> (8, 8, ndf*4)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (8, 8, ndf*4) -> (4, 4, ndf*8)
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
		self.output = nn.Sequential(
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
		)

	def forward(self, x):
		x = self.main_module(x)
		return self.output(x)

	def feature_extraction(self, x):
		x = self.main_module(x)
		return x.view(-1, ndf*8*4*4)


class WGAN_GP():
	def __init__(self):
		self.G = Generator()
		self.D = Discriminator()

		self.learing_rate = lr
		self.b1 = beta1
		self.b2 = 0.999
		self.batch_size = batch_size

		self.d_optimizer = optim.Adam(self.D.parameters(), lr = self.learing_rate, betas=(self.b1, self.b2))
		self.g_optimizer = optim.Adam(self.G.parameters(), lr = self.learing_rate, betas=(self.b1, self.b2))

		self.batch_size = batch_size
		self.critic_iter = n_critic

	def train(self):
		# 创建dataset
		dataset = datasets.ImageFolder(root=param.dataroot,
									   transform=transforms.Compose([
										   transforms.Resize(image_size),
										   transforms.CenterCrop(image_size),
										   transforms.ToTensor(),
										   # transforms.Normalize((0.5,), (0.5,)),
									   ]))
		# 创建dataloader
		dataloader = torch.utils.data.DataLoader(dataset,
												 batch_size=batch_size,
												 shuffle=True,
												 num_workers=works)
