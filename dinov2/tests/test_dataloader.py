import pytest
import torch
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import time
import logging

from dinov2.configs import dinov2_default_config
from dinov2.train.setup import setup_dataloader_multi_resolution

logger = logging.getLogger("dino")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg_x = OmegaConf.load("dinov2/configs/tests/multires.yaml")
    cfg_x = OmegaConf.merge(default_cfg, cfg_x)
    return cfg_x


@pytest.fixture
def dataloader(cfg):
    inputs_dtype = torch.bfloat16
    return setup_dataloader_multi_resolution(cfg, inputs_dtype, 0)


def test_dataloader_speed(cfg):
    num_workers = cfg.train.num_workers
    logger.info(f"Using {num_workers} workers.")

    inputs_dtype = torch.bfloat16
    t0 = time.time()
    dataloader = setup_dataloader_multi_resolution(cfg, inputs_dtype, 0)
    dataloader_iter = iter(dataloader)
    tf = time.time() - t0
    logger.info(f"Created dataloader in {tf:.06f} seconds.")

    for idx in range(1):
        t0 = time.time()
        data = next(dataloader_iter)
        tf = time.time() - t0
        logger.info(f"Batch {idx}: waited {tf:.06f} seconds.")
        time.sleep(0.6)


def test_dataloader_output(dataloader):
    def extract_imgs(crops, output_path, idx):
        _, D, W, H = crops.shape

        plt.figure()
        plt.imshow(crops[0, D // 2, :, :], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}A.png"))

        plt.figure()
        plt.imshow(crops[0, :, W // 2, :], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}B.png"))

        plt.figure()
        plt.imshow(crops[0, :, :, H // 2], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}C.png"))

    dataloader_iter = iter(dataloader)
    output_path = "dinov2/tests/out"

    for idx, data in enumerate(dataloader_iter):

        global_view = data["collated_global_crops"].float().numpy()

        extract_imgs(global_view, output_path, f"g_{idx:02}")

        local_view = data["collated_local_crops"].float().numpy()

        extract_imgs(local_view, output_path, f"l_{idx:02}")

        if idx == 10:
            break
