from random import choice, randint


class TripletMixin:
    """
    Mixin class for loading triplets on __get_item__ for any Dataset. Must be used with a torch.Dataset subclass,
    as it assumes the existence of the `classes`, `class_to_idx` and `imgs` fields.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with class keys, for efficient online
        triplet generation.
        """
        super().__init__(*args, **kwargs)
        self.imgs_dict = {
            idx: [index for index, img in enumerate(self.imgs) if img[1] == idx]
            for cls, idx in self.class_to_idx.items()
        }

    def __get_random_item__(self, cls):
        """
        Get a random item from the specified class.
        :param cls: the class idx
        :type: int
        :return: an item tuple
        :type: tuple
        """
        class_image_indexes = self.imgs_dict[cls]
        return super().__getitem__(class_image_indexes[randint(0, len(class_image_indexes) - 1)])

    def __getitem__(self, index):
        """
        Return a triplet consisting of an anchor (the indexed item), a positive (a random example of a different class),
        and a negative (a random example of the same class).
        :param index: an item's index
        :type: int
        :return: a tuple with the anchor
        :type: tuple(torch.Tensor, list<torch.Tensor>, int)
        """
        positive_class = self.imgs[index][1]
        negative_classes = list(range(0, positive_class)) + list(range(positive_class + 1, len(self.classes)))
        anchor = super().__getitem__(index)
        positive = self.__get_random_item__(positive_class)
        negative = self.__get_random_item__(choice(negative_classes))
        return anchor, positive, negative