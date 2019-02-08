from settings import DATA_SETS

from src.trainers.sketchy_cnn import train_sketchy_cnn


if __name__ == '__main__':
    train_sketchy_cnn(DATA_SETS['sketchy_test']['images'], DATA_SETS['sketchy_test']['dimensions'][0])
