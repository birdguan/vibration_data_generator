import torch
from src.parameters import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import autograd
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main_module = nn.Sequential(
			# 噪声 -> (4, 4, ngf*8)
			nn.ConvTranspose2d(nz, ngf*32, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf*32),
			nn.ReLU(True),

			# (4, 4, ngf*8) -> (8, 8, ngf*4)
			nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*16),
			nn.ReLU(True),

			# (8, 8, ngf*4) -> (16, 16, ngf*2)
			nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*8),
			nn.ReLU(True),

			# (16, 16, ngf*2) -> (32, 32, ngf*2)
			nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*4),
			nn.ReLU(True),

			# (32, 32, ngf*2) -> (64, 64, ngf)
			nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*2),
			nn.ReLU(True),

			# (64, 64, ngf) -> (128, 128, ngf)
			nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
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
			nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(ndf*2, affine=True),
			nn.LeakyReLU(0.2, inplace=True),

			# (64, 64, ndf) -> (32, 32, ndf*2)
			nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(ndf*4, affine=True),
			nn.LeakyReLU(0.2, inplace=True),

			# (32, 32, ndf*2) -> (16, 16, ndf*2)
			nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(ndf*8, affine=True),
			nn.LeakyReLU(0.2, inplace=True),

			# (16, 16, ndf*2) -> (8, 8, ndf*4)
			nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(ndf*16, affine=True),
			nn.LeakyReLU(0.2, inplace=True),

			# (8, 8, ndf*4) -> (4, 4, ndf*8)
			nn.Conv2d(ndf*16, ndf*32, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(ndf*32, affine=True),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.output = nn.Sequential(
			nn.Conv2d(ndf*32, 1, 4, 1, 0, bias=False),
		)

	# def forward(self, x):
	# 	x = self.main_module(x)
	# 	return self.output(x)

	def forward(self, x):
		x = self.main_module(x)
		return x.view(-1, ndf*32*4*4)


class WGAN_GP():
	def __init__(self):
		self.device = torch.device("cuda: 0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
		self.G = Generator().to(self.device)
		self.D = Discriminator().to(self.device)

		self.learing_rate = lr
		self.b1 = beta1
		self.b2 = 0.999
		self.batch_size = batch_size
		self.lambda_term = 10
		self.criterion = nn.BCELoss()
		self.BCE_stable = nn.BCEWithLogitsLoss()
		self.fixed_noise = torch.rand(16, nz, 1, 1, device=self.device)
		self.real_label = 1
		self.fake_label = 0

		self.d_optimizer = optim.Adam(self.D.parameters(), lr = self.learing_rate, betas=(self.b1, self.b2))
		self.g_optimizer = optim.Adam(self.G.parameters(), lr = self.learing_rate, betas=(self.b1, self.b2))

		self.batch_size = batch_size
		self.critic_iter = n_critic


	def calculate_gradient_penalty(self, real, fake, batch_size):
		eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
		eta = eta.expand(batch_size, real.size(1), real.size(2), real.size(3))
		eta = eta.to(self.device)
		interpolated = eta * real + ((1 - eta) * fake).to(self.device)
		interpolated = Variable(interpolated, requires_grad=True)
		prob_interpolated = self.D(interpolated)
		gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
							 grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
								  create_graph=True,
								  retain_graph=True)[0]
		grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
		return grad_penalty


	def save_model(self, epoch):
		torch.save(self.G.state_dict(), "../frozen_model/wgan-gp-generator_{}.pkl".format(epoch))
		torch.save(self.D.state_dict(), "../frozen_model/wgan-gp-discriminator_{}.pkl".format(epoch))

	def load_model(self, epoch):
		self.D.load_state_dict(torch.load("../frozen_model/wgan-gp-generator_{}.pkl".format(epoch)))
		self.G.load_state_dict(torch.load("../frozen_model/wgan-gp-discriminator_{}.pkl".format(epoch)))

	def train(self):
		# 创建dataset
		dataset = datasets.ImageFolder(root=dataroot,
									   transform=transforms.Compose([
										   transforms.Resize(image_size),
										   transforms.CenterCrop(image_size),
										   transforms.ToTensor(),
										   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
									   ]))
		# 创建dataloader
		dataloader = torch.utils.data.DataLoader(dataset,
												 batch_size=self.batch_size,
												 shuffle=True,
												 num_workers=works)
		G_losses = []
		D_losses = []
		img_list = []
		fixed_noise = torch.rand(16, nz, 1, 1, device=self.device)
		fig = plt.figure(figsize=(4, 4))
		plt.ion()
		plt.axis("off")
		iters = 0
		one = torch.FloatTensor([1]).to(self.device)
		mone = one * -1
		print(" ==== START TRAINING ==== ")
		for epoch in range(num_epochs):
			for i, data in enumerate(dataloader, 0):
				##############################
				# 更新鉴别器
				##############################
				for d_iter in range(self.critic_iter):
					# real batch
					self.D.zero_grad()
					real = data[0].to(self.device)
					errD_real = self.D(real).mean()
					errD_real.backward(mone)

					batch_size = real.size(0)
					pred_real = self.D(real).view(-1)
					D_x = pred_real.mean().item()

					# fake batch
					noise = torch.randn(batch_size, nz, 1, 1, device=self.device)
					fake = self.G(noise)
					errD_fake = self.D(fake.detach()).mean()
					errD_fake.backward(one)
					pred_fake = self.D(fake.detach()).view(-1)
					D_G_z1 = pred_fake.mean().item()

					# gradient penalty
					gradient_penalty = self.calculate_gradient_penalty(real, fake.detach(), batch_size)
					gradient_penalty.backward(retain_graph=True)
					# gradient_penalty = 0

					errD = errD_real + errD_fake + gradient_penalty
					self.d_optimizer.step()

				##############################
				# 更新生成器
				##############################
				self.G.zero_grad()
				noise = torch.randn(batch_size, nz, 1, 1, device=self.device)
				fake = self.G(noise)
				errG = self.D(fake).mean()
				errG.backward(mone)
				self.g_optimizer.step()

				if i % 50 == 0:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
						  % (epoch, num_epochs, i, len(dataloader),
							 errD.item(), errG.item(), D_x, D_G_z1))

				G_losses.append(errG.item())
				D_losses.append(errD.item())
				iters += 1

			# with torch.no_grad():
			# 	fake = self.G(self.fixed_noise).detach().cpu()
			# img_list = vutils.make_grid(fake, nrow=4, padding=2, normalize=True)
			# plt.imshow(np.transpose(img_list, (1, 2, 0)))
			# plt.title("epoch:" + str(epoch))
			# plt.pause(1)
			# plt.clf()

			# save model and sampling images every 5 epochs
			if epoch % 5 == 0:
				self.save_model(epoch)
				if not os.path.exists('../training/'):
					os.mkdir('../training/')
				samples = self.G(fixed_noise)
				# samples = samples.mul(0.5).add(0.5).detach().cpu()
				# grid = vutils.make_grid(samples, nrow=4)
				vutils.save_image(samples, filename='../training/img_generated_epoch_{}.png'.format(epoch),
								  nrow=4)



		plt.ioff()


if __name__ == '__main__':

	model = WGAN_GP()
	model.train()