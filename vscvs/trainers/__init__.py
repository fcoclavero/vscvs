from .cnn import train_cnn
from .gan import train_gan_bimodal, train_gan_bimodal_siamese, train_gan_multimodal
from .gcn import train_classification_gcn, train_hog_gcn
from .resnet import train_resnet
from .resnext import train_resnext
from .siamese import train_siamese_cnn, train_siamese_resnet, train_siamese_resnext
from .triplet import train_triplet_cnn, train_triplet_resnet, train_triplet_resnext
