import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_SETS = {
    'sketchy': {
        'root': r'C:\Users\Chopan\Documents\Data\sketchy',
        'photos': r'C:\Users\Chopan\Documents\Data\sketchy\photo\tx_000000000000', # 12500
        'sketches': r'C:\Users\Chopan\Documents\Data\sketchy\sketch\tx_000000000000', # 75481
        'classes': r'C:\Users\Chopan\Documents\Data\sketchy\classes.pickle',
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy_test': {
        'root': r'C:Users\Chopan\Documents\Data\sketchy_test',
        'photos': r'C:\Users\Chopan\Documents\Data\sketchy_test\photo', # 1250
        'sketches': r'C:\Users\Chopan\Documents\Data\sketchy_test\sketch', # 1250
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
