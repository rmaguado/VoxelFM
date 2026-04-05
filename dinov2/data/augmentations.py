# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random
from typing import List, Tuple

import torch
from omegaconf import DictConfig

from dinov2.data.transforms import *


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        local_crops_number,
        local_crops_size,
        local_crops_scale,
        global_crops_size,
        global_crops_scale,
        mean,
        std,
        range_lower,
        range_upper,
    ) -> None:
        """
        Initializes an instance of the Augmentations class.

        Args:
            config (DictConfig): The primary configuration object.
            dataset_config (DictConfig): The dataset configuration object.
        """
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.local_crops_scale = local_crops_scale

        self.global_crops_size = global_crops_size
        self.global_crops_scale = global_crops_scale

        self.norm = Norm(
            mean=mean,
            std=std,
            vmin=range_lower,
            vmax=range_upper,
        )

        self.global1 = ImageTransforms()
        self.global1 += RandomCrop3D(
            size=self.global_crops_size, scale=self.global_crops_scale
        )
        self.global1 += Flip()
        self.global1 += Window(p=0.5)
        self.global1 += self.norm

        self.global2 = ImageTransforms()
        self.global2 += RandomCrop3D(
            size=self.global_crops_size, scale=self.global_crops_scale
        )
        self.global2 += Flip()
        self.global2 += self.norm

        self.local1 = ImageTransforms()
        self.local1 += RandomCrop3D(
            size=self.local_crops_size, scale=self.local_crops_scale
        )
        self.local1 += self.norm

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """
        Apply augmentations to the input image.

        Args:
            image: The input image to apply augmentations to.

        Returns:
            output: A dictionary containing the augmented image crops.
        """
        output = {}

        global_crop_1 = self.global1(image)
        global_crop_2 = self.global2(image)

        output["global_crops"] = [global_crop_1, global_crop_2]

        local_crops = [self.local1(image) for _ in range(self.local_crops_number)]
        output["local_crops"] = local_crops

        output["global_crops_size"] = self.global_crops_size

        return output


class DataAugmentationDINO2p5D(object):
    def __init__(
        self,
        local_crops_number: int,
        local_crops_size: int | Tuple[int, int, int],
        local_crops_scale: Tuple[float, float],
        global_crops_size: int | Tuple[int, int, int],
        global_crops_scale: Tuple[float, float],
        mean: float,
        std: float,
        range_lower: float,
        range_upper: float,
        slab_thickness: int,
        slab_axis: int | None = None,
    ) -> None:
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.local_crops_scale = local_crops_scale
        self.global_crops_size = global_crops_size
        self.global_crops_scale = global_crops_scale
        self._fixed_slab_axis = slab_axis  # None = random per sample

        self.slab_sampler = SlabSampler(
            slab_thickness=slab_thickness,
            slab_axis=slab_axis,
        )

        self._global_crop = SlabAwareRandomCrop3D(
            size=self.global_crops_size,
            scale=self.global_crops_scale,
            slab_axis=2,
        )
        self._local_crop = SlabAwareRandomCrop3D(
            size=self.local_crops_size,
            scale=self.local_crops_scale,
            slab_axis=2,
        )

        self.norm = Norm(mean=mean, std=std, vmin=range_lower, vmax=range_upper)

        self.global1_post = ImageTransforms()
        self.global1_post += Flip()
        self.global1_post += Window(p=0.5)
        self.global1_post += self.norm

        self.global2_post = ImageTransforms()
        self.global2_post += Flip()
        self.global2_post += self.norm

        self.local1_post = ImageTransforms()
        self.local1_post += self.norm

    def _apply_crop(
        self, crop_transform: SlabAwareRandomCrop3D, slab: torch.Tensor
    ) -> torch.Tensor:
        """Temporarily set the slab axis on the crop transform and apply it."""
        return crop_transform(slab)

    def __call__(self, image) -> dict:

        slab = self.slab_sampler(image)

        output = {
            "global_crops": [
                self.global1_post(self._apply_crop(self._global_crop, slab)),
                self.global2_post(self._apply_crop(self._global_crop, slab)),
            ],
            "local_crops": [
                self.local1_post(self._apply_crop(self._local_crop, slab))
                for _ in range(self.local_crops_number)
            ],
            "global_crops_size": self.global_crops_size,
        }

        return output
