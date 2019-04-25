class TripletMixin:
    def __getitem__(self, index):
        """
        Return a triplet consisting of an anchor (the indexed item), a positive (a random example of a different class),
        and a negative (a random example of the same class).
        :param index: an item's index
        :type: int
        :return: a tuple with the anchor
        :type: tuple(torch.Tensor, list<torch.Tensor>, int)
        """
        photo, cls, name = self.photos_dataset[index]
        return photo, torch.stack([self.sketch_dataset[i][0] for i in self.__sketches__[index]]), cls