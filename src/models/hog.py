import torch

from hog.histogram import gradient, magnitude_orientation, hog, visualise_histogram


def img_2_hog(image):
    im = image[0][0].numpy()
    h = hog(im, cell_size=(8, 8), cells_per_block=(1, 1), visualise=False, nbins=9, signed_orientation=False, normalise=True)
    return torch.from_numpy(h).reshape(-1)