import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, output_dimension, feature_depth, n_channels=3, n_gpu=0):
        """
        Convolutional network that creates feature vectors for images.
        :param output_dimension: the size of the output vector (common vector space dimensionality)
        :type: int
        :param n_channels: number of channels of input images
        :type: int
        :param feature_depth: regulates the size of channels in the convolutional layers
        :type: int
        :param n_gpu: number of available gpus for training
        :type: int
        :return the image encoder pytorch model
        :type: pytorch.nn.Module
        """
        super(ImageEncoder, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is (n_channels) x 256 x 256
            nn.Conv2d(n_channels, feature_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth) x 128 x 128
            nn.Conv2d(feature_depth, feature_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth*2) x 64 x 64
            nn.Conv2d(feature_depth * 2, feature_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth*4) x 32 x 32
            nn.Conv2d(feature_depth * 4, feature_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth*8) x 16 x 16
            nn.Conv2d(feature_depth * 8, feature_depth * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth*16) x 8 x 8
            nn.Conv2d(feature_depth * 16, feature_depth * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_depth*32) x 4 x 4
            nn.Conv2d(feature_depth * 32, output_dimension, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)