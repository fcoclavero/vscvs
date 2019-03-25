import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_SETS = {
    'sketchy': {
        'root': 'H:\\Data\\Sketchy\\',
        'images': 'H:\\Data\\Sketchy\\rendered_256x256\\256x256\\photo\\tx_000000000000',
        'sketches': 'H:\\Data\\Sketchy\\rendered_256x256\\256x256\\sketch\\tx_000000000000',
        'classes': 'H:\\Data\\Sketchy\\classes.pickle',
        'dimensions': (256, 256),
        'language': 'en'
    },
    'sketchy_test': {
        'root': 'C:\\Users\\Chopan\\Documents\\Data\\sketchy_test\\',
        'images': 'C:\\Users\\Chopan\\Documents\\Data\\sketchy_test\\photo',
        'sketches': 'C:\\Users\\Chopan\\Documents\\Data\\sketchy_test\\sketch',
        'classes': 'C:\\Users\\Chopan\\Documents\\Data\\sketchy_test\\classes.pickle',
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
