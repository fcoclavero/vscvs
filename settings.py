import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

CHECKPOINT_NAME_FORMAT = '%y-%m-%dT%H-%M'

DATA_DIR = r'C:\Users\Chopan\Dropbox\Data'

DATA_SOURCES = {
    'sketchy': {
        'root': os.path.join(DATA_DIR, r'sketchy'),
        'photos': os.path.join(DATA_DIR, r'sketchy\photo\tx_000000000000'), # 12500
        'sketches': os.path.join(DATA_DIR, r'sketchy\sketch\tx_000000000000'), # 75481
        'classes': os.path.join(DATA_DIR, r'sketchy\classes.pickle'),
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy_test': {
        'root': os.path.join(DATA_DIR, r'C:Users\Chopan\Documents\Data\sketchy_test'),
        'photos': os.path.join(DATA_DIR, r'sketchy_test\photo'), # 1250
        'sketches': os.path.join(DATA_DIR, r'sketchy_test\sketch'), # 1250
        'classes': os.path.join(DATA_DIR, r'sketchy_test\classes.pickle'),
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sample_vectors': {
        'pickle': os.path.join(ROOT_DIR, 'static', 'pickles', 'discriminators', 'sample-vectors.pickle'),
        'dimensions': (100000, 100)
    },
    'sample_vectors_onehot': {
        'pickle': os.path.join(ROOT_DIR, 'static', 'pickles', 'discriminators', 'sample-vectors-onehot.pickle'),
        'dimensions': (100000, 100)
    }
}
