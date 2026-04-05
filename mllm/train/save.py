import deepspeed
import torch
import os


def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with deepspeed.zero.GatheredParameters(list(model.parameters())):

        if deepspeed.comm.get_rank() == 0:  # type: ignore

            if hasattr(model.lm, "peft_config"):
                lora_dir = os.path.join(output_dir, "lora_adapters_dir")
                os.makedirs(lora_dir, exist_ok=True)
                model.lm.save_pretrained(lora_dir)

            torch.save(
                model.mm_projector.state_dict(),
                os.path.join(output_dir, "mm_projector.pth"),
            )

            config_dir = os.path.join(output_dir, "base_config_dir")
            os.makedirs(config_dir, exist_ok=True)
            model.lm.config.save_pretrained(config_dir)
