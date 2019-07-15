import sys

sys.path.append("..")

import torch

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

from hog.histogram import gradient, magnitude_orientation, hog, visualise_histogram

from settings import DATA_SOURCES
from src.datasets.sketchy import Sketchy

# Load the Dataset class
dataset = Sketchy(DATA_SOURCES['sketchy']['sketches'])

# Number of workers for dataloader
workers = 8

# Batch size for dataloader
batch_size = 1

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(torch.cuda.is_available())

# Plot some training images
image = dataset[0]
plt.figure(figsize=(4,4))
plt.axis("off")
plt.title("Class: %s" % image[1])
plt.imshow(np.transpose(vutils.make_grid(image[0].to(device), padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()