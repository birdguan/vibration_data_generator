import src.model as model
import src.parameters as param
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML



def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

# 创建dataset
dataset = datasets.ImageFolder(root=param.dataroot,
							   transform=transforms.Compose([
								   transforms.Resize(param.image_size),
								   transforms.CenterCrop(param.image_size),
								   transforms.ToTensor(),
								   # transforms.Normalize((0.5,), (0.5,)),
							   ]))
# 创建dataloader
dataloader = torch.utils.data.DataLoader(dataset,
										 batch_size=param.batch_size,
										 shuffle=True,
										 num_workers=param.works)
# 选择设备
device = torch.device("cuda: 0" if (torch.cuda.is_available() and param.ngpu > 0) else "cpu")

netG = model.Generator(param.ngpu).to(device)
# netG.apply(weights_init)
netD = model.Discriminator(param.ngpu).to(device)
# netD.apply(weights_init)

criterion = nn.BCELoss()
BCE_stable = nn.BCEWithLogitsLoss()
fixed_noise = torch.rand(16, param.nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimizerG = optim.Adam(netG.parameters(), lr=param.lr, betas=(param.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=param.lr, betas=(param.beta1, 0.999))


# Train Loop
def train():
	G_losses = []
	D_losses = []
	img_list = []
	fig = plt.figure(figsize=(4, 4))
	plt.ion()
	plt.axis("off")
	iters = 0
	print(" ==== START TRAINING ==== ")
	for epoch in range(param.num_epochs):
		for i, data in enumerate(dataloader, 0):
			##############################
			# 更新鉴别器
			##############################

			# real batch
			netD.zero_grad()
			real = data[0].to(device)
			batch_size = real.size(0)
			pred_real_label = torch.full((batch_size,), real_label, device=device)
			pred_real = netD(real).view(-1)
			D_x = pred_real.mean().item()

			# fake batch
			noise = torch.randn(batch_size, param.nz, 1, 1, device=device)
			fake = netG(noise)
			pred_fake_label = torch.full((batch_size,), fake_label, device=device)
			pred_fake = netD(fake.detach()).view(-1)
			D_G_z1 = pred_fake.mean().item()
			errD = (BCE_stable(pred_real - torch.mean(pred_fake), pred_real_label)
					 + BCE_stable(pred_fake - torch.mean(pred_real), pred_fake_label))/2
			errD.backward()
			optimizerD.step()

			##############################
			# 更新生成器
			##############################
			if iters % param.n_critic == 0:
				netG.zero_grad()
				pred_fake = netD(fake).view(-1)
				errG = (BCE_stable(pred_real.detach() - torch.mean(pred_fake), pred_fake_label) +
						BCE_stable(pred_fake - torch.mean(pred_real.detach()), pred_real_label))/2
				errG.backward()
				D_G_z2 = pred_fake.mean().item()
				optimizerG.step()


			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, param.num_epochs, i, len(dataloader),
						 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

			G_losses.append(errG.item())
			D_losses.append(errD.item())
			iters += 1

		with torch.no_grad():
			fake = netG(fixed_noise).detach().cpu()
		img_list = vutils.make_grid(fake, nrow=4, padding=2, normalize=True)
		plt.imshow(np.transpose(img_list, (1, 2, 0)))
		plt.title("epoch:" + str(epoch))
		plt.pause(1)
		plt.clf()
	plt.ioff()
		# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
		# HTML(ani.to_jshtml())


if __name__ == '__main__':
	train()