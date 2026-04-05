import torch

from dinov2.models import DinoVisionTransformer


def get_autocast_dtype(config) -> torch.dtype:
    teacher_dtype_str = config.compute_precision
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model(path_to_checkpoint, config, device):
    args = config.student

    vit_kwargs = dict(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        init_values=args.layerscale,
        rope_base=args.rope_base,
        rope_rescale_coords=args.rope_rescale_coords,
        ffn_layer=args.ffn_layer,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
    )

    model = DinoVisionTransformer(**vit_kwargs)

    state_dict = torch.load(path_to_checkpoint)["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(p) for p in ["dino_head", "ibot_head"])
    }
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    return model
