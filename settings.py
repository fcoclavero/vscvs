import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_SETS = {
    'sketchy': {
        'root': r'C:\Users\Chopan\Documents\Data\sketchy',
        'images': r'C:\Users\Chopan\Documents\Data\sketchy\photo\tx_000000000000',
        'sketches': r'C:\Users\Chopan\Documents\Data\sketchy\tx_000000000000',
        'classes': r'C:\Users\Chopan\Documents\Data\sketchy\classes.pickle',
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy_test': {
        'root': r'C:Users\Chopan\Documents\Data\sketchy_test',
        'images': r'C:\Users\Chopan\Documents\Data\sketchy_test\photo',
        'sketches': r'C:\Users\Chopan\Documents\Data\sketchy_test\sketch',
        'classes': r'C:\Users\Chopan\Documents\Data\sketchy_test\classes.pickle',
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
