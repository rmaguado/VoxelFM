import os
import pathlib
import torch
import torch.distributed as dist

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.configs import load_config
from mllm.models.vl_model import VisionLanguageModel
from mllm.models.injectors import ImageTokenInjector
from mllm.models.projectors import AttentionalPoolProjector
from mllm.train.lora import configure_lora
from mllm.train.trainer import LLaVATrainer
from mllm.train.save import save_model
from mllm.data import make_datasets


def train():
    print("Starting Vision-Language training")

    model_args, data_args, training_args = load_config()

    local_rank = training_args.local_rank
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    dtype = (
        torch.bfloat16
        if training_args.bf16
        else torch.float16 if training_args.fp16 else torch.float32
    )

    lm = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        attn_implementation=(
            "flash_attention_2"
            if getattr(training_args, "flash_attention", False)
            else None
        ),
    )

    lm.requires_grad_(False)
    lm.config.use_cache = False

    mm_projector = AttentionalPoolProjector(
        embed_dim=model_args.mm_vision_hidden_size,
        hidden_dim=model_args.mm_projector_hidden_size,
        output_dim=lm.config.hidden_size,
        resample_tokens=model_args.image_tokens,
    )

    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing")

        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            lm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.transformers_cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lm.config.tokenizer_model_max_length = tokenizer.model_max_length

    if model_args.lora.enable:
        print("Enabling LoRA")
        lm = configure_lora(lm, model_args)

    for p in mm_projector.parameters():
        p.requires_grad = True

    if model_args.pretrain_checkpoint_path is not None:
        ckpt = model_args.pretrain_checkpoint_path
        print(f"Loading pretrained checkpoint from {ckpt}")

        adapters_path = os.path.join(ckpt, "lora_adapters_dir")
        if os.path.exists(adapters_path):
            print("Loading LoRA adapters")
            lm = PeftModel.from_pretrained(lm, adapters_path)

        projector_path = os.path.join(ckpt, "mm_projector.pth")
        if os.path.exists(projector_path):
            print("Loading projector weights")
            state = torch.load(projector_path, map_location="cpu")
            mm_projector.load_state_dict(state, strict=True)

    injector = ImageTokenInjector()

    model = VisionLanguageModel(lm=lm, mm_projector=mm_projector, injector=injector)

    model.to(dtype=dtype)

    output_dir = str(training_args.output_dir)
    run_dir = os.path.dirname(output_dir)
    with open(os.path.join(run_dir, "trainable_params.txt"), "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}\n")

    data_module = make_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = LLaVATrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    resume_from_checkpoint = list(pathlib.Path(output_dir).glob("checkpoint-*"))

    if resume_from_checkpoint:
        print("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.lm.config.use_cache = True  # type: ignore

    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    save_model(model, final_dir)

    print("Training complete")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
