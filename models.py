import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, action_dim=7):
        super(CNN,self).__init__()

        self.action_dim = action_dim
        # 3 x 64 x 64
        self.conv1=nn.Conv2d(3,16,5,padding=2) # (16,32,32)
        self.conv2=nn.Conv2d(16,32,5,padding=2) # (32,16,16)
        self.conv3=nn.Conv2d(32,64,3,padding=1) # (64,8,8)
        self.conv4=nn.Conv2d(64,128,3,padding=1) # (128,4,4)
        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,64)
        # self.fc3 = nn.Linear(64,6)
        self.fc3 = nn.Linear(64,self.action_dim)  # add stop action
        # self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2,2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out,2,2)
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2, 2)
        out = out.view(in_size,-1) # flatten
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        # out = self.dropout(out)
        out = self.fc3(out)

        return out


noise_dim = 100
latent_dim = 64
ngf = 32
ndf = 32


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()  # WGAN don't need sigmoid
        )

    def forward(self, input):
        return self.main(input)


class Predictor(nn.Module):
    def __init__(self, classes_num=10, action_dim=7):
        super(Predictor, self).__init__()

        self.classes_num = classes_num
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 4 x 4 x 4 = latent_dim
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim+64, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.fc = nn.Sequential(
            nn.Linear(self.action_dim+self.classes_num, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, action, task):
        info = torch.cat([task, action], 1)
        embedding = self.fc(info)
        encoded = self.encoder(x).view(x.shape[0], -1)
        encoded = torch.cat([encoded, embedding], 1)
        encoded = encoded.view(x.shape[0], -1, 1, 1)
        decoded = self.decoder(encoded)
        return decoded
