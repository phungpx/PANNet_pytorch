from typing import List, Tuple, Optional

from abstract import BaseDataLoader
from .PAN_dataset import PANDataset


class PANDataLoader(BaseDataLoader):
    """PAN data loading demo using BaseDataLoader
    """
    def __init__(
        self,
        dirnames: List[str],
        imsize: int = 640,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        shrink_ratio: float = 0.5,
        image_extents: List[str] = ['.jpg'],
        label_extent: str = '.json',
        transforms: Optional[list] = None,
        require_transforms: Optional[list] = None,
        validation_split: float = 0.0,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = False,
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
