# 数据集根目录
dataroot = "../imgs"

# dataloader worker数
works = 2

# batch size
batch_size = 64

# 图像大小
image_size = 256

# 通道数
nc = 3

# 噪声向量大小
nz = 1000

# 生成器特征图大小
ngf = 64

# 鉴别器特征图大小
ndf = 64

# epoch
num_epochs = 500

# 学习率
lr = 0.0002

# beta1
beta1 = 0.5

# GPU数量，0表示使用CPU
ngpu = 1

#
n_critic = 5
