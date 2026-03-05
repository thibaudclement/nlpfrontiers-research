from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple
import torch
from transformers import PreTrainedModel

@dataclass
class TrainableParameterSummary:
    total_parameters: int
    trainable_parameters: int
    trainable_fraction: float

# Count total and trainable parameters (for sanity-check logging)
def summarize_trainable_parameters(model: PreTrainedModel) -> TrainableParameterSummary:
    total_parameters = 0
    trainable_parameters = 0

    for parameter in model.parameters():
        num = int(parameter.numel())
        total_parameters += num
        if parameter.requires_grad:
            trainable_parameters += num

    trainable_fraction = float(trainable_parameters / total_parameters) if total_parameters > 0 else 0.0
    return TrainableParameterSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_fraction=trainable_fraction,
    )

# Set requires_grad for all model parameters to provided value
def set_requires_grad_for_all_parameters(model: PreTrainedModel, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = bool(requires_grad)

# Extract BERT encoder layer modules (standard Hugging Face BERT-style QA models)
def get_bert_encoder_layers(model: PreTrainedModel) -> List[torch.nn.Module]:
    # Target BertForQuestionAnswering to expose `model.bert.encoder.layer`
    if not hasattr(model, "bert"):
        raise ValueError("Model does not have attribute 'bert'. This modifier currently supports BERT-style models only.")
    bert_module = getattr(model, "bert")

    if not hasattr(bert_module, "encoder") or not hasattr(bert_module.encoder, "layer"):
        raise ValueError("Model does not have 'bert.encoder.layer'. Cannot apply BERT layer freezing.")
    return list(bert_module.encoder.layer)

# Ensure the QA head parameters remain trainable
def set_question_answering_head_trainable(model: PreTrainedModel, requires_grad: bool = True) -> None:
    # For BertForQuestionAnswering the head is typically `qa_outputs`
    if hasattr(model, "qa_outputs"):
        for parameter in getattr(model, "qa_outputs").parameters():
            parameter.requires_grad = bool(requires_grad)

# Freeze all BERT encoder layers except those specified in trainable_layer_indices
def set_trainable_bert_encoder_layers(
    model: PreTrainedModel,
    trainable_layer_indices: Set[int],
) -> Tuple[int, List[int]]:
    encoder_layers = get_bert_encoder_layers(model)
    number_of_layers = len(encoder_layers)

    # Normalize and validate indices
    normalized_trainable_indices: Set[int] = set()
    for index in trainable_layer_indices:
        if index < 0 or index >= number_of_layers:
            raise ValueError(f"Trainable layer index {index} is out of range for {number_of_layers} layers.")
        normalized_trainable_indices.add(int(index))

    # Freeze all encoder layers first
    for layer in encoder_layers:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    # Unfreeze only requested layers
    for index in sorted(normalized_trainable_indices):
        for parameter in encoder_layers[index].parameters():
            parameter.requires_grad = True

    return number_of_layers, sorted(normalized_trainable_indices)

# Freeze bottom k encoder layers (lowest indices) and leave others trainable
def freeze_bottom_k_bert_encoder_layers(model: PreTrainedModel, number_of_bottom_layers_to_freeze: int) -> Tuple[int, int]:
    encoder_layers = get_bert_encoder_layers(model)
    number_of_layers = len(encoder_layers)

    k = int(number_of_bottom_layers_to_freeze)
    if k < 0 or k > number_of_layers:
        raise ValueError(f"number_of_bottom_layers_to_freeze must be in [0, {number_of_layers}], got {k}.")

    for layer_index, layer in enumerate(encoder_layers):
        requires_grad = layer_index >= k
        for parameter in layer.parameters():
            parameter.requires_grad = bool(requires_grad)

    return number_of_layers, k

# Apply freezing strategy specified by config keys
def apply_bert_freezing_strategy_from_config(model: PreTrainedModel, training_config: dict) -> str:
    freeze_strategy = str(training_config.get("freeze_strategy", "none")).strip().lower()

    if freeze_strategy == "none":
        # Default: full fine-tune.
        set_requires_grad_for_all_parameters(model, True)
        return "freeze_strategy=none (full fine-tune)"

    if freeze_strategy == "freeze_bottom_k":
        # Freeze bottom K, train remaining and head
        k = int(training_config.get("freeze_bottom_k", 0))
        number_of_layers, used_k = freeze_bottom_k_bert_encoder_layers(model, number_of_bottom_layers_to_freeze=k)
        set_question_answering_head_trainable(model, True)
        return f"freeze_strategy=freeze_bottom_k (k={used_k}, total_layers={number_of_layers})"

    if freeze_strategy == "train_first_last":
        # Train only specified layer indices, freeze all others, keep head trainable.
        indices = training_config.get("trainable_layer_indices", None)
        if indices is None:
            raise ValueError("train_first_last requires 'trainable_layer_indices' in training config.")
        trainable_set = {int(x) for x in indices}
        number_of_layers, used_indices = set_trainable_bert_encoder_layers(model, trainable_layer_indices=trainable_set)
        set_question_answering_head_trainable(model, True)
        return f"freeze_strategy=train_first_last (trainable_layers={used_indices}, total_layers={number_of_layers})"

    raise ValueError(f"Unknown freeze_strategy='{freeze_strategy}'. Expected one of: none, freeze_bottom_k, train_first_last.")