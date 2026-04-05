"""
Copied and modified from:
https://github.com/2U1/Llama3.2-Vision-Finetune/blob/master/src/training/train.py#L73
"""

import torch
from peft import LoraConfig, get_peft_model


def find_target_linear_names(
    model, exclude_modules, include_modules=None, must_have_keywords=None
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if must_have_keywords is not None:
            if not all(inc_keyword in name for inc_keyword in must_have_keywords):
                continue
        if any(ex_keyword in name for ex_keyword in exclude_modules):
            continue
        if include_modules is not None:
            if not any(inc_keyword in name for inc_keyword in include_modules):
                continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    return lora_module_names


def configure_lora(model, model_args):
    lora_args = model_args.lora
    target_modules = find_target_linear_names(
        model,
        exclude_modules=lora_args.exclude_modules,
        include_modules=lora_args.modules,
    )

    config = LoraConfig(
        target_modules=target_modules,
        r=lora_args.rank,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, config)
