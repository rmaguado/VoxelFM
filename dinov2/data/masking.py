# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
from typing import Optional, Tuple


class MaskingGenerator3D:
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        if not isinstance(input_size, tuple) or len(input_size) != 3:
            raise ValueError(
                "input_size must be a tuple of 3 integers (depth, height, width)"
            )
        self.depth, self.height, self.width = input_size

        self.num_patches = self.depth * self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        if num_masking_patches is not None:
            self.max_num_patches = num_masking_patches
        else:
            assert max_num_patches is not None
            self.max_num_patches = max_num_patches

        if max_aspect is None:
            max_aspect = 1 / min_aspect

        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator3D(D=%d, H=%d, W=%d, aspect_ratios=(%.3f, %.3f)." % (
            self.depth,
            self.height,
            self.width,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[0],
        )
        return repr_str

    def get_shape(self):
        return self.depth, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_volume = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio_h_w = math.exp(random.uniform(*self.log_aspect_ratio))
            aspect_ratio_d_w = math.exp(random.uniform(*self.log_aspect_ratio))

            w = int(
                round(
                    (target_volume / (aspect_ratio_h_w * aspect_ratio_d_w)) ** (1 / 3)
                )
            )
            h = int(round(w * aspect_ratio_h_w))
            d = int(round(w * aspect_ratio_d_w))

            if (
                w > 0
                and h > 0
                and d > 0
                and w < self.width
                and h < self.height
                and d < self.depth
            ):
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                front = random.randint(0, self.depth - d)

                num_masked = mask[
                    front : front + d, top : top + h, left : left + w
                ].sum()

                if 0 < d * h * w - num_masked <= max_mask_patches:
                    for i in range(front, front + d):
                        for j in range(top, top + h):
                            for k in range(left, left + w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        if mask.sum() >= num_masking_patches:
            return mask

        shape = mask.shape
        m2 = mask.flatten()

        num_to_add = num_masking_patches - m2.sum()
        unmasked_indices = np.where(~m2)[0]

        num_to_add = min(num_to_add, len(unmasked_indices))

        to_add = np.random.choice(unmasked_indices, size=num_to_add, replace=False)
        m2[to_add] = True
        return m2.reshape(shape)
