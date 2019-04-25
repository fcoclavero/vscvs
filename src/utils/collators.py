import torch


def sketchy_collate(batch):
    """
    Custom collate_fn, used to pack a batch of dataset items. By default, torch stacks the input items
    to from a tensor of size N*C*H*W, so every item in the batch must have the same dimensions.
    Given that we stack sketches for each photo (in __get_item__), and the number of sketches is variable,
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