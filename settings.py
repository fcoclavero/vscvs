__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Project settings and parameters. """


import os

from dotenv import load_dotenv


load_dotenv()

try:
    ROOT_DIR = os.environ['ROOT_DIR']
except KeyError:
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

CHECKPOINT_NAME_FORMAT = '%y-%m-%dT%H-%M'

DATA_DIR = os.environ['DATA_DIR']

DATA_SOURCES = {
    'sketchy-photos': {
        'root': os.path.join(DATA_DIR, 'sketchy'),
        'images': os.path.join(DATA_DIR, 'sketchy', 'photo', 'tx_000000000000'), # 12500
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy-sketches': {
        'root': os.path.join(DATA_DIR, 'sketchy'),
        'images': os.path.join(DATA_DIR, 'sketchy', 'sketch', 'tx_000000000000'), # 75481
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy-test-photos': {
        'root': os.path.join(DATA_DIR, 'sketchy_test'),
        'images': os.path.join(DATA_DIR, 'sketchy_test', 'photo'), # 1250
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy-test-sketches': {
        'root': os.path.join(DATA_DIR, 'sketchy_test'),
        'images': os.path.join(DATA_DIR, 'sketchy_test', 'sketch'), # 1250
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sample_vectors': {
        'pickle': os.path.join(DATA_DIR, 'sample-vectors.pickle'),
        'dimensions': (100000, 100)
    },
    'sample_vectors_one-hot': {
        'pickle': os.path.join(DATA_DIR, 'sample-vectors-one-hot.pickle'),
        'dimensions': (100000, 100)
    }
}
