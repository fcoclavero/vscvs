__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom collator functions for customizing DataLoaders. """


import torch


def sketchy_mixed_collate(batch):
    """
    Custom collate_fn, used to pack a batch of dataset items. By default, torch stacks the input items
    to from a tensor of size N*C*H*W, so every item in the batch must have the same dimensions.
    Given that we stack sketches for each photo (in __getitem__), and the number of sketches is variable,
    we have to use a custom collate_fn.
    :param batch: list of Dataset items with length equal to the batch size.
    :type: list<Tuple>
    :return: a prepared batch, as it will be handed to the trainer
    :type: Tuple<torch.Tensor, list<torch.Tensor>, torch.Tensor>
    """
    photos = [item[0] for item in batch]
    sketches = [item[1] for item in batch]
    classes = [item[2] for item in batch]
    return torch.stack(photos), sketches, torch.Tensor(classes)


class triplet_collate:
    """
    Collate function factory that collates the anchor, positive and negative elements of a triplet using the
    provided collate function for individual items.
    Ignite pickles the collate function, so we can't simply use a function factory function that receives the individual
    collate function and returns a locally defined collate function that receives batches, as there is no way to
    address them by name.
    Thus, we must create a serializable object that can be used as a function. To do this, we implement the `__call__`
    function, that makes the object callable, just like a function.
    [Reference](https://stackoverflow.com/a/12022055)
    """
    def __init__(self, collate_fn):
        """
        Triplet collate function factory constructor. Saves the individual item collate function.
        :param collate_fn: collate function for individual items
        :type: function
        """
        self.collate_fn = collate_fn

    def __call__(self, batch):
        anchor = [item[0] for item in batch]
        positive = [item[1] for item in batch]
        negative = [item[2] for item in batch]
        return self.collate_fn(anchor), self.collate_fn(positive), self.collate_fn(negative)