# model.py

import torch 
import torch.nn as nn 


class Discrimiator(nn.Module):
    def __init__(self,  num_channels, maps):
        super(Discrimiator, self).__init__()
        # print(maps)
        self.net = nn.Sequential(
                # INPUT : N x C x 64 x 64
                nn.Conv2d(in_channels = num_channels, out_channels=maps, kernel_size=4, stride=2, padding=1, bias = False), # N x C x 32 x 32
                nn.LeakyReLU(0.2, inplace=True),
                self._block(in_channels=maps, out_channels=maps*2, kernel_size=4 , stride=2, padding=1),      # N x C x 16 x 16
                self._block(in_channels=maps*2, out_channels=maps*4, kernel_size=4 , stride=2, padding=1),    # N x C x 8 x 8
                self._block(in_channels=maps*4, out_channels=maps*8, kernel_size=4 , stride=2, padding=1),    # N x C x 4 x 4 
                nn.Conv2d(in_channels=maps*8, out_channels=1, kernel_size=4 , stride=1, padding=0, bias = False),           # N x 1 x 1 x 1     
                nn.Sigmoid()
        )

    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.net(x)



class Generator(nn.Module):
    def __init__(self, noise_channels ,img_channels, maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input  = N x 100 x 1 x 1 
                self._block(in_channels=noise_channels, out_channels=maps*8, kernel_size=4 , stride=1, padding=0),    # N x C x 4 x 4
                self._block(in_channels=maps*8, out_channels=maps*4, kernel_size=4 , stride=2, padding=1),    # N x C x 16 x 16
                self._block(in_channels=maps*4, out_channels=maps*2, kernel_size=4 , stride=2, padding=1),    # N x C x 32 x 32
                self._block(in_channels=maps*2, out_channels=maps*1, kernel_size=4 , stride=2, padding=1),    # N x C x 32 x 32


                nn.ConvTranspose2d(in_channels = maps*1, out_channels=img_channels, kernel_size=4, stride=2, padding=1, bias = False), # N x C x 64 x 64
                nn.Tanh()
        )

    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)



def initialize_wieghts(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_discremenator():
    x = torch.rand(32, 3, 64, 64)
    disc = Discrimiator(num_channels=3, maps = 32)
    print(disc(x).shape)


def test_generator():
    x = torch.rand(32, 100,1, 1)
    disc = Generator(noise_channels=100, img_channels = 3, maps = 32)
    print(disc(x).shape)


if __name__ == '__main__':
    test_discremenator()
    test_generator()