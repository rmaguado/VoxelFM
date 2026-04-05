import torch
import torch.nn as nn

from .injectors import ImageTokenInjector
from .projectors import AttentionalPoolProjector


class VisionLanguageModel(nn.Module):
    injector: ImageTokenInjector
    mm_projector: AttentionalPoolProjector

    def __init__(self, lm, injector, mm_projector):
        super().__init__()
        self.lm = lm
        self.injector = injector
        self.mm_projector = mm_projector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        image_features=None,
        **kwargs,
    ):
        if image_features is None:
            return self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        device = self.lm.device
        dtype = self.lm.dtype

        image_features = [
            feat.to(device=device, dtype=dtype) if feat is not None else None
            for feat in image_features
        ]

        image_embeds = self.mm_projector(image_features)
        inputs_embeds, labels, attention_mask = self.injector(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            embed_tokens=self.lm.get_input_embeddings(),
            dtype=dtype,
        )

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self, input_ids=None, attention_mask=None, image_features=None, **kwargs
    ):
        if image_features is None:
            return self.lm.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

        device = self.lm.device
        dtype = self.lm.dtype

        image_features = [
            feat.to(device=device, dtype=dtype) for feat in image_features
        ]

        image_embeds = self.mm_projector(image_features)

        inputs_embeds, _, attention_mask = self.injector(
            input_ids=input_ids,
            labels=None,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            embed_tokens=self.lm.get_input_embeddings(),
            dtype=dtype,
        )

        return self.lm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.lm.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        """Check if gradient checkpointing is enabled."""
        if self.lm.is_gradient_checkpointing:
            return self.lm.is_gradient_checkpointing
        return False
