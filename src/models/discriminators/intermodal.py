import torch.nn as nn


class InterModalDiscriminator(nn.Module):
    def __init__(self, input_dimension, n_gpu):
        """
        Fully connected network that classifies vectors in the common vector space as belonging any mode
        (image or sketch).
        :param input_dimension: the size of the input vector (common vector space dimensionality)
        :type: int
        :param n_gpu: number of available gpus for training
        :type: int
        :return the image encoder pytorch model
        :type: pytorch.nn.Module
        """
        super(InterModalDiscriminator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # state size. nz
            nn.Linear(input_dimension, 75),
            # state size. 75
            nn.ReLU(),
            nn.Linear(75, 25),
            # state size. 25
            nn.ReLU(),
            nn.Linear(25, 2), # binary one-hot encoding vector
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.main(input)