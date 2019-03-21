def split(data, split_proportion = 0.8):
    """
    Splits incoming data into two sets, one for training and one for tests.
    Current implementation just slices on the index corresponding to the given proportion. This could be changed
    to a random, class balanced version.
    :param data: the dataset to be split
    :type: indexed object
    :param split_proportion:
    :return: the two resulting datasets
    :type: indexed object
    """
    test_index = int(len(data) * split_proportion)
    return data[:test_index], data[test_index:]
