import os
import torch
import math
from typing import Optional

from torch.utils.data import Dataset, Sampler
from torch.optim.lr_scheduler import LambdaLR

from transformers import Trainer
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from mllm.train.save import save_model
from mllm.data.samplers import LengthGroupedSampler


class LLaVATrainer(Trainer):

    def _get_train_sampler(
        self, train_dataset: Optional[Dataset] = None
    ) -> Optional[Sampler]:
        assert train_dataset is not None and hasattr(train_dataset, "modality_lengths")
        lengths = train_dataset.modality_lengths  # type: ignore
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            lengths=lengths,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()  # type: ignore
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()  # type: ignore
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None  # type: ignore
    ):
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        min_lr = self.args.min_lr  # type: ignore
        base_lr = self.args.learning_rate
        self._created_lr_scheduler = True

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                warmup_percent_done = current_step / float(max(1, num_warmup_steps))
                lr = min_lr + (base_lr - min_lr) * warmup_percent_done
            else:
                progress = (current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = min_lr + (base_lr - min_lr) * cosine_decay
            return lr / base_lr

        return LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, model, trial):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = str(self._get_output_dir(trial=trial))
        save_output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(save_output_dir, exist_ok=True)

        save_model(model, save_output_dir)

        super()._save_checkpoint(model, trial)
