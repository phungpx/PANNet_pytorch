from abstract import BaseDataLoader
from .PAN_dataset import PANDataset


class PANDataLoader(BaseDataLoader):
    """PAN data loading demo using BaseDataLoader
    """
    def __init__(
        self,
        dirnames, imsize, mean, std, shrink_ratio,
        image_extents, label_extent,
        transforms, require_transforms,
        validation_split=0.0,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    ):
        self.dataset = PANDataset(
            dirnames, imsize, mean, std, shrink_ratio,
            image_extents, label_extent,
            transforms, require_transforms
        )
        super(PANDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            validation_split=validation_split,
        )
