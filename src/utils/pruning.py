from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import PreTrainedModel

@dataclass
class SparsityReport:
    total_weight_parameters: int
    total_zero_parameters: int
    global_sparsity: float

# Collect (module, parameter_name) tuples to prune
def collect_prunable_parameters(
    model: PreTrainedModel,
    prune_linear_layers: bool,
    prune_question_answering_head: bool,
) -> List[Tuple[nn.Module, str]]:
    prunable_parameters: List[Tuple[nn.Module, str]] = []

    # Collect Linear layer weights across whole model
    if prune_linear_layers:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Optionally exclude QA head from pruning
                if (not prune_question_answering_head) and ("qa_outputs" in module_name):
                    continue
                prunable_parameters.append((module, "weight"))

    return prunable_parameters

# Compute global sparsity across all Linear weights (and optionally QA head) after pruning
def compute_global_weight_sparsity(
    model: PreTrainedModel,
    prune_linear_layers: bool,
    prune_question_answering_head: bool,
) -> SparsityReport:
    total_weight_parameters = 0
    total_zero_parameters = 0

    # Count zeros across selected weights
    for module_name, module in model.named_modules():
        if prune_linear_layers and isinstance(module, nn.Linear):
            if (not prune_question_answering_head) and ("qa_outputs" in module_name):
                continue
            weight_tensor = module.weight.detach()
            total_weight_parameters += int(weight_tensor.numel())
            total_zero_parameters += int((weight_tensor == 0).sum().item())

    global_sparsity = float(total_zero_parameters / total_weight_parameters) if total_weight_parameters > 0 else 0.0
    return SparsityReport(
        total_weight_parameters=total_weight_parameters,
        total_zero_parameters=total_zero_parameters,
        global_sparsity=global_sparsity,
    )


# Apply global unstructured magnitude pruning to selected parameters
def apply_global_magnitude_pruning(
    model: PreTrainedModel,
    target_sparsity: float,
    prune_linear_layers: bool = True,
    prune_question_answering_head: bool = True,
    remove_pruning_reparameterization: bool = True,
) -> Dict[str, Any]:
    # Validate pruning fraction
    if target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")

    # Collect parameters to prune
    prunable_parameters = collect_prunable_parameters(
        model=model,
        prune_linear_layers=prune_linear_layers,
        prune_question_answering_head=prune_question_answering_head,
    )
    if len(prunable_parameters) == 0:
        raise ValueError("No prunable parameters were found. Check pruning configuration.")

    # Compute sparsity before pruning
    sparsity_before = compute_global_weight_sparsity(
        model=model,
        prune_linear_layers=prune_linear_layers,
        prune_question_answering_head=prune_question_answering_head,
    )

    # Apply global magnitude pruning across all selected weights
    prune.global_unstructured(
        prunable_parameters,
        pruning_method=prune.L1Unstructured,
        amount=float(target_sparsity),
    )

    # Optionally remove pruning reparameterization to make zeros permanent
    if remove_pruning_reparameterization:
        for module, parameter_name in prunable_parameters:
            prune.remove(module, parameter_name)

    # Compute sparsity after pruning
    sparsity_after = compute_global_weight_sparsity(
        model=model,
        prune_linear_layers=prune_linear_layers,
        prune_question_answering_head=prune_question_answering_head,
    )

    return {
        "target_sparsity": float(target_sparsity),
        "pruned_parameter_tuples": len(prunable_parameters),
        "sparsity_before": {
            "global_sparsity": sparsity_before.global_sparsity,
            "total_weight_parameters": sparsity_before.total_weight_parameters,
            "total_zero_parameters": sparsity_before.total_zero_parameters,
        },
        "sparsity_after": {
            "global_sparsity": sparsity_after.global_sparsity,
            "total_weight_parameters": sparsity_after.total_weight_parameters,
            "total_zero_parameters": sparsity_after.total_zero_parameters,
        },
    }