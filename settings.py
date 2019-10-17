__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Project settings and parameters. """


import os

from dotenv import load_dotenv


# Load env
load_dotenv()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

CHECKPOINT_NAME_FORMAT = '%y-%m-%dT%H-%M'

DATA_DIR = os.environ['DATA_DIR']

DATA_SOURCES = {
    'sketchy': {
        'root': os.path.join(DATA_DIR, 'sketchy'),
        'photos': os.path.join(DATA_DIR, 'sketchy', 'photo', 'tx_000000000000'), # 12500
        'sketches': os.path.join(DATA_DIR, 'sketchy', 'sketch', 'tx_000000000000'), # 75481
        'classes': os.path.join(DATA_DIR, 'sketchy', 'classes.pickle'),
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy_test': {
        'root': os.path.join(DATA_DIR, 'sketchy_test'),
        'photos': os.path.join(DATA_DIR, 'sketchy_test', 'photo'), # 1250
        'sketches': os.path.join(DATA_DIR, 'sketchy_test', 'sketch'), # 1250
        'classes': os.path.join(DATA_DIR, 'sketchy_test', 'classes.pickle'),
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sample_vectors': {
        'pickle': os.path.join(ROOT_DIR, 'data', 'pickles', 'discriminators', 'sample-vectors.pickle'),
        'dimensions': (100000, 100)
    },
    'sample_vectors_onehot': {
        'pickle': os.path.join(ROOT_DIR, 'data', 'pickles', 'discriminators', 'sample-vectors-onehot.pickle'),
        'dimensions': (100000, 100)
    }
}
