import os
import torch
from typing import List, Dict, Tuple, Optional, Callable
import polars as pl
import numpy as np
import logging


logger = logging.getLogger("dinov2")


Spacing = Tuple[float, ...]


class VolumeDataset:
    """
    Base class for volumetric medical image datasets.

    Handles loading of dataset metadata, resampling of images to isotropic spacing,
    and application of transformations. Subclasses should implement the get_image_data method.

    Args:
        dataset_name (str): Name of the dataset.
        index_path (str): Path to the CSV file containing dataset index/metadata.
        modality (str): Imaging modality (e.g., 'ct', 'mri').
        transform (Callable): Transformation function to apply to the loaded volume.
    """

    def __init__(
        self,
        dataset_name: str,
        index_path: str,
        modality: str,
        transforms: Callable,
        global_crops_size: List[int],
        bounds: Tuple[float, float] = (-1000, 1900),
    ) -> None:
        self.dataset_name = dataset_name
        self.df = pl.read_csv(index_path)
        self.modality = modality
        self.transforms = transforms
        self.global_crops_size = global_crops_size
        self.bounds = bounds

        self._check_df()

    def __len__(self) -> int:
        return len(self.df)

    def _check_df(self):
        raise NotImplementedError

    def get_image_data(self, idx: int) -> Tuple[torch.Tensor, Optional[Spacing]]:
        raise NotImplementedError

    def __getitem__(self, idx: int):

        image = self.get_image_data(idx)

        return self.transforms(image)


class TorchVolumeDataset(VolumeDataset):
    """
    Dataset class for loading volumetric image saved as torch pth.

    Inherits from VolumeDataset and implements get_image_data for torch pth.
    """

    def _check_df(self):
        len_init = len(self.df)
        path_exists = pl.Series("exists", [os.path.exists(p) for p in self.df["path"]])

        df_exists = self.df.filter(path_exists)
        len_exists = len(df_exists)

        min_required_size = int(max(self.global_crops_size) / 2)

        def is_valid_shape(shape_str):
            try:
                dims = [int(d) for d in shape_str.strip("[]").split(",")]
                return all(d >= min_required_size for d in dims)
            except:
                return False

        df_valid = pl.Series(
            "valid", [is_valid_shape(p) for p in df_exists["shape_final"]]
        )

        df_final = df_exists.filter(df_valid)
        len_final = len(df_final)

        if len_exists < len_init:
            logger.warning(
                f"Dropped {len_init - len_exists} samples due to file not found (dataset: {self.dataset_name})."
            )
        if len_final < len_exists:
            logger.warning(
                f"Dropped {len_exists - len_final} samples due to inappropriate size (dataset: {self.dataset_name})."
            )

        self.df = df_final.select("path")

    def get_image_data(self, idx: int):
        file_path = self.df[int(idx), "path"]
        image_array = torch.load(file_path, mmap=True)

        return image_array


class MultiDataset:
    """
    Collates multiple datasets into a single dataset interface.

    Allows indexing across multiple datasets as if they were a single dataset.

    Args:
        datasets (list): List of dataset objects to combine.
    """

    def __init__(self, datasets: list) -> None:
        """
        Initializes a MultiDataset object for collating various dataset objects.

        Args:
            datasets (list): A list of datasets.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        """
        Returns the total number of samples across all datasets.

        Returns:
            int: Total number of samples.
        """
        return self.cumulative_sizes[-1]

    def get_dataset_sizes(self) -> List[int]:
        """
        Returns the sizes of each individual dataset.

        Returns:
            List[int]: List of dataset sizes.
        """
        return [len(d) for d in self.datasets]

    def get_dataset_names(self) -> List[str]:
        """
        Returns the names of each individual dataset.

        Returns:
            List[str]: List of dataset names.
        """
        return [d.dataset_name for d in self.datasets]

    def _find_dataset_idx(self, idx: int) -> int:
        """
        Finds which dataset a global index belongs to.

        Args:
            idx (int): Global index.

        Returns:
            int: Index of the dataset in the datasets list.
        """
        return int(np.searchsorted(self.cumulative_sizes, idx, side="right"))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the appropriate dataset based on the global index.

        Args:
            index (int): Global index across all datasets.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of transformed image views.
        """
        dataset_idx = self._find_dataset_idx(index)

        if dataset_idx > 0:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            dataset_index = index

        return self.datasets[dataset_idx][dataset_index]
