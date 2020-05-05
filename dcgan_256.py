from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import wgan_gp_preprocess_256_3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DCGAN():
    def __init__(self):
        # 输入维度
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        self.n_generate = 2 #训练次数对比

        optimizer_d = Adam(0.0001, 0.5)
        optimizer_g = Adam(0.0002, 0.5)

        # 判别器
        self.discrimnator = self.build_discriminator()
        self.discrimnator.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])

        # 生成器
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # for the combined model we will only tarin the generator
        self.discrimnator.trainable = False

        # the discrimnator takes generated images as input and determines validity
        valid = self.discrimnator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(8 * 8 * 256, input_shape=(self.latent_dim,), use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((8, 8, 256)))
        """
            反卷积
        """
        model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='SAME', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())  #  8 * 8 *256 -> 16*16*128


        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='SAME', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())  #   16*16*128 -> 32 * 32 * 64


        model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='SAME', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())  #  32 * 32 * 64 -> 64*64*32

        model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='SAME', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())  #    64*64*32-> 128*128*16

        # 128*128*16 -> 256*256*3
        model.add(Conv2DTranspose(self.channels, (3, 3), strides=(2, 2), padding='SAME', use_bias=False))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)


    def build_discriminator(self):
        model = Sequential()
        # 256*256*3   -> 128*128*32
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # 128*128*32   ->  64*64*32
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # 64*64*32   ->  32*32*32
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # 32*32*32  ->  `16*16*16
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # `16*16*16 ->  8*8*8
        model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self,epochs,batch_size = 128,save_interval = 50):
        #load the dataset

        X_train = wgan_gp_preprocess_256_3.prepro()

        print(X_train.shape)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)


        #Adversial ground truths
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
                #train the Discrimator
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0,1,(batch_size,self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discrimnator.train_on_batch(imgs,valid)
            d_loss_fake = self.discrimnator.train_on_batch(gen_imgs,fake)
            d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
            #train the generator user the combined model fix the discriminator

            g_loss = self.combined.train_on_batch(noise,valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # if at interval ->save generated image smaples
            if epoch %save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/dcgan_256_3_%d.png" % epoch)
        plt.close()


    def sample_images_one(self,epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print(gen_imgs.shape)#1 256 256 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.imshow(gen_imgs[0,:,:,0])
        plt.axis('off')
        plt.savefig("images/dcgan_256*256*1_%d.png" % epoch)
        plt.close()




if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)























