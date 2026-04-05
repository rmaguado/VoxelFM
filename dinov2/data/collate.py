# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

from typing import List, Tuple, Dict, Callable, Any


def collate_data_and_cast(
    samples_list: List[Any],
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    n_tokens: int,
    mask_generator: Callable[[int], Any],
) -> Dict[str, Any]:
    """
    Collates data and casts it to the specified data type.

    Args:
        samples_list (list): A list of samples.
        mask_ratio_tuple (tuple): A tuple containing the minimum and maximum mask ratios.
        mask_probability (float): The probability of masking.
        dtype (torch.dtype): The data type to cast the collated crops to.
        n_tokens (int): The number of tokens.
        mask_generator (function): A function that generates masks.

    Returns:
        dict: A dictionary containing the collated data.
            - collated_global_crops (torch.Tensor): The collated global crops.
            - collated_local_crops (torch.Tensor): The collated local crops.
            - collated_masks (torch.Tensor): The collated masks.
            - mask_indices_list (torch.Tensor): The mask indices list.
            - masks_weight (torch.Tensor): The masks weight.
            - upperbound (int): The upperbound.
            - n_masked_patches (torch.Tensor): The number of masked patches.
    """
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack(
        [s["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )

    collated_local_crops = torch.stack(
        [s["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i].item()
        prob_max = probs[i + 1].item()
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))
            )
        )
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
