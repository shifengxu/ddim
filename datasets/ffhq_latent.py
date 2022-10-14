import numpy as np
from typing import Callable, Optional, Any
from torchvision.datasets import DatasetFolder as DataFolder
from torchvision.datasets.folder import default_loader


class FFHQ_Latent(DataFolder):
    def __init__(
        self,
        root: str,
        extensions=('vq', 'npy'),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = np.load(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target
