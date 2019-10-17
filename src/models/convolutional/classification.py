__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image classification convolutional models. """


import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 256x256x3
        self.convolution_1 = nn.Conv2d(3, 6, 5) # 252x252x6
        # 126x126x6
        self.convolution_2 = nn.Conv2d(6, 16, 5) # 122x122x16
        # 61x61x16
        self.convolution_3 = nn.Conv2d(16, 20, 4) # 58x58x20
        # 29x29x20
        self.fully_connected_1 = nn.Linear(20 * 29 * 29, 15000) # 15000
        # 15000
        self.fully_connected_2 = nn.Linear(15000, 1200) # 1200

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.convolution_1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.convolution_2(x)), 2)
        x = F.max_pool2d(F.relu(self.convolution_3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fully_connected_1(x))
        x = F.relu(self.fully_connected_2(x))
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ClassificationConvolutionalNetwork(ConvolutionalNetwork):
    def __init__(self):
        super().__init__()
        # 1200
        self.fully_connected_3 = nn.Linear(1200, 125) # 125

    def forward(self, x):
        x = super().forward(x)
        x = self.fully_connected_3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    @property
    def embedding_network(self):
        """
        Convert the classification CNN into an embedding network by eliminating the last layer.
        The network weights are loaded from the Model's state_dict and copied to the returned model. If used with a
        trained network, the returned embedding network should produce image descriptors.
        :return: the derived embedding network
        :type: ConvolutionalNetwork
        """
        state_dict = self.state_dict()  # contains network layers
        del state_dict['fully_connected_3.weight']  # delete last layer weights
        del state_dict['fully_connected_3.bias']  # and bias
        model = ConvolutionalNetwork()  # new model object with same structure as checkpoint, but without last layer
        incorrect_keys = model.load_state_dict(state_dict, strict=False)  # load the trained model weights into it
        if not sum([len(keys) for keys in incorrect_keys]):  # no incompatibility between checkpoint and selected model
            return model
        else:  # log incompatibility
            raise ValueError(
                ''.join(['{}: {}\n'.format(field, incorrect_keys[i]) for i, field in enumerate(incorrect_keys._fields)])
            )
