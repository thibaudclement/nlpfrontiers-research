from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import Trainer

@dataclass
class TrainingCounters:
    number_of_training_steps: int = 0
    number_of_training_examples: int = 0
    number_of_training_tokens: int = 0

class CountingTrainer(Trainer):
    # Initialize Trainer that counts steps/examples/tokens during training
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_counters = TrainingCounters()

    # Override training_step to count units before executing step
    def training_step(self, model, inputs):
        # Count training steps (note: counts micro-steps when using gradient accumulation)
        self.training_counters.number_of_training_steps += 1

        # Count examples from the batch size
        if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
            self.training_counters.number_of_training_examples += int(inputs["input_ids"].shape[0])

        # Count tokens as the number of non-padding tokens in the attention mask
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                self.training_counters.number_of_training_tokens += int(attention_mask.sum().item())

        return super().training_step(model, inputs)