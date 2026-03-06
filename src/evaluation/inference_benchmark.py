from __future__ import annotations
import inspect
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from src.data.squad_v2 import prepare_squad_v2_evaluation_features, postprocess_squad_v2_predictions
from src.energy.meter import EnergyMeter
from src.evaluation.squad_metrics import compute_squad_v2_metrics
from src.utils.io import append_line_to_text_file, write_json_file
from src.utils.token_count import count_non_padding_tokens_in_feature_dataset
from src.models.counting_trainer import CountingTrainer

# Load the raw SQuAD v2 evaluation split only
def load_raw_squad_v2_evaluation_split(
    huggingface_dataset_id: str,
    evaluation_split_name: str,
    maximum_evaluation_examples: Optional[int] = None,
):
    raw_evaluation_split = load_dataset(huggingface_dataset_id, split=evaluation_split_name)

    # Optionally truncate the evaluation split for fast debugging
    if maximum_evaluation_examples is not None:
        raw_evaluation_split = raw_evaluation_split.select(range(int(maximum_evaluation_examples)))

    return raw_evaluation_split

# Clamp document stride for shorter sequence lengths to avoid tokenizer errors
def clamp_document_stride_for_sequence_length(configured_document_stride: int, maximum_sequence_length: int) -> int:
    return int(min(int(configured_document_stride), max(1, int(maximum_sequence_length) // 2)))

# Filter batch fields so they match the model forward signature
def filter_batch_for_model_forward(model: PreTrainedModel, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    supported_argument_names = set(inspect.signature(model.forward).parameters.keys())
    return {name: value for name, value in batch.items() if name in supported_argument_names}

# Build evaluation DataLoader with dynamic padding
def build_evaluation_dataloader(
    tokenized_evaluation_features_for_model,
    tokenizer: PreTrainedTokenizerBase,
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    pad_to_multiple_of: Optional[int],
) -> DataLoader:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    return DataLoader(
        tokenized_evaluation_features_for_model,
        batch_size=int(per_device_evaluation_batch_size),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=int(dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
    )

# Run short warmup pass before measured inference
def run_inference_warmup(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    number_of_warmup_batches: int,
) -> None:
    if int(number_of_warmup_batches) <= 0:
        return

    model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= int(number_of_warmup_batches):
                break

            batch_on_device = {name: tensor.to(device) for name, tensor in batch.items()}
            batch_for_model = filter_batch_for_model_forward(model=model, batch=batch_on_device)
            _ = model(**batch_for_model)

    if device.type == "cuda":
        torch.cuda.synchronize()

# Run measured inference with Hugging Face Trainer.predict for consistency with the rest of the harness
def run_measured_question_answering_inference_with_trainer(
    run_directory: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_evaluation_features_for_model,
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    pad_to_multiple_of: Optional[int],
    device: torch.device,
) -> Dict[str, Any]:
    # Use the same dynamic padding policy as the training / evaluation harness
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Build minimal Trainer arguments for prediction-only benchmarking
    trainer_arguments = TrainingArguments(
        output_dir=str(run_directory / "huggingface_trainer_predict"),
        per_device_eval_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        report_to=[],
    )

    # Construct Trainer arguments in a version-compatible way
    trainer_init_signature = inspect.signature(CountingTrainer.__init__)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": trainer_arguments,
        "data_collator": data_collator,
    }
    if "tokenizer" in trainer_init_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = CountingTrainer(**trainer_kwargs)

    # Measure wall-clock runtime around Trainer.predict
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start_time_seconds = time.perf_counter()
    prediction_output = trainer.predict(tokenized_evaluation_features_for_model)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_end_time_seconds = time.perf_counter()

    raw_predictions = prediction_output.predictions
    if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
        start_logits, end_logits = raw_predictions
    else:
        start_logits, end_logits = raw_predictions

    return {
        "start_logits": np.array(start_logits),
        "end_logits": np.array(end_logits),
        "inference_runtime_seconds": float(inference_end_time_seconds - inference_start_time_seconds),
    }

# Normalize precision mode names and check for hardware / software support
def normalize_precision_mode_name(precision_mode: str) -> str:
    normalized = str(precision_mode).strip().lower()

    aliases = {
        "float32": "fp32",
        "fp32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float8": "fp8",
        "fp8": "fp8",
    }

    if normalized not in aliases:
        raise ValueError(f"Unsupported precision mode: {precision_mode}")

    return aliases[normalized]

# Get GPU compute capability
def get_gpu_compute_capability(device: torch.device) -> Optional[tuple[int, int]]:
    if device.type != "cuda":
        return None

    major, minor = torch.cuda.get_device_capability(device=device)
    return int(major), int(minor)

# Check if specified precision mode is supported on current hardware and software configuration
def check_precision_mode_support(
    precision_mode: str,
    device: torch.device,
) -> tuple[bool, Optional[str]]:
    normalized_precision_mode = normalize_precision_mode_name(precision_mode)

    if normalized_precision_mode == "fp32":
        return True, None

    if device.type != "cuda":
        return False, f"{normalized_precision_mode} inference requires CUDA in this harness"

    if normalized_precision_mode == "fp16":
        return True, None

    if normalized_precision_mode == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return True, None
        return False, "bf16 is not supported by the current CUDA device / PyTorch build"

    if normalized_precision_mode == "fp8":
        float8_dtypes_present = (
            hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "float8_e5m2")
        )
        if not float8_dtypes_present:
            return False, "PyTorch float8 dtypes are not available in this build"

        compute_capability = get_gpu_compute_capability(device=device)
        if compute_capability is None:
            return False, "fp8 requires CUDA hardware support"

        major, minor = compute_capability
        if (major, minor) < (9, 0):
            return False, "fp8 inference typically requires Hopper-class GPU support"

        return False, (
            "fp8 is not enabled in this harness because standard Hugging Face BERT "
            "inference does not reliably support end-to-end fp8 execution without "
            "specialized kernels"
        )

    raise ValueError(f"Unsupported precision mode: {precision_mode}")

# Map normalized precision mode to corresponding PyTorch dtype
def get_model_parameter_dtype_for_precision_mode(
    precision_mode: str,
) -> torch.dtype:
    normalized_precision_mode = normalize_precision_mode_name(precision_mode)

    if normalized_precision_mode == "fp32":
        return torch.float32
    if normalized_precision_mode == "fp16":
        return torch.float16
    if normalized_precision_mode == "bf16":
        return torch.bfloat16

    raise ValueError(
        f"Precision mode {precision_mode} cannot be mapped to a model parameter dtype"
    )

# Cast model parameters to appropriate dtype
def cast_model_for_inference_precision(
    model: PreTrainedModel,
    precision_mode: str,
) -> PreTrainedModel:
    target_dtype = get_model_parameter_dtype_for_precision_mode(precision_mode)
    return model.to(dtype=target_dtype)

# Normalize token pruning keep-ratio values for stable labeling and folder names
def normalize_token_pruning_keep_ratio(keep_ratio: float) -> float:
    normalized_keep_ratio = float(keep_ratio)
    if normalized_keep_ratio <= 0.0 or normalized_keep_ratio > 1.0:
        raise ValueError(f"Token pruning keep_ratio must be in (0, 1], got {keep_ratio}")
    return normalized_keep_ratio

# Format token pruning keep-ratio for logs, plots, and subdirectory names
def format_token_pruning_keep_ratio_label(keep_ratio: float) -> str:
    normalized_keep_ratio = normalize_token_pruning_keep_ratio(keep_ratio)
    return f"{normalized_keep_ratio:.2f}"

# Build a map from example id to raw example fields used during token pruning
def build_raw_example_lookup_by_id(raw_evaluation_split) -> Dict[str, Dict[str, Any]]:
    raw_example_lookup_by_id: Dict[str, Dict[str, Any]] = {}

    for raw_example in raw_evaluation_split:
        raw_example_lookup_by_id[str(raw_example["id"])] = {
            "id": str(raw_example["id"]),
            "question": raw_example["question"],
            "context": raw_example["context"],
        }

    return raw_example_lookup_by_id

# Identify question token positions
def get_question_token_positions_for_feature(
    feature: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    input_ids = feature["input_ids"]
    attention_mask = feature["attention_mask"]
    offset_mapping = feature["offset_mapping"]
    token_type_ids = feature.get("token_type_ids")
    special_token_ids = set(tokenizer.all_special_ids)

    if token_type_ids is not None:
        return [
            index
            for index, (token_id, token_type_id, attention_value, offset_value) in enumerate(
                zip(input_ids, token_type_ids, attention_mask, offset_mapping)
            )
            if int(attention_value) == 1
            and int(token_type_id) == 0
            and token_id not in special_token_ids
            and offset_value is None
        ]

    context_positions = [
        index
        for index, (attention_value, offset_value) in enumerate(zip(attention_mask, offset_mapping))
        if int(attention_value) == 1 and offset_value is not None
    ]

    first_context_position = min(context_positions) if len(context_positions) > 0 else len(input_ids)

    return [
        index
        for index, (token_id, attention_value, offset_value) in enumerate(
            zip(input_ids, attention_mask, offset_mapping)
        )
        if index < first_context_position
        and int(attention_value) == 1
        and token_id not in special_token_ids
        and offset_value is None
    ]

# Identify context token positions
def get_context_token_positions_for_feature(feature: Dict[str, Any]) -> list[int]:
    input_ids = feature["input_ids"]
    attention_mask = feature["attention_mask"]
    offset_mapping = feature["offset_mapping"]

    return [
        index
        for index, (_, attention_value, offset_value) in enumerate(
            zip(input_ids, attention_mask, offset_mapping)
        )
        if int(attention_value) == 1 and offset_value is not None
    ]

# Score context tokens
def score_context_tokens_with_question_embedding_similarity(
    feature: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    embedding_weight_cpu: torch.Tensor,
) -> Dict[int, float]:
    input_ids = feature["input_ids"]

    question_token_positions = get_question_token_positions_for_feature(
        feature=feature,
        tokenizer=tokenizer,
    )
    context_token_positions = get_context_token_positions_for_feature(feature=feature)

    if len(context_token_positions) == 0:
        return {}

    if len(question_token_positions) == 0:
        return {position: 0.0 for position in context_token_positions}

    question_input_ids = torch.tensor(
        [int(input_ids[position]) for position in question_token_positions],
        dtype=torch.long,
    )
    question_embeddings = embedding_weight_cpu[question_input_ids]
    question_embedding_mean = question_embeddings.mean(dim=0)

    question_embedding_mean_norm = torch.linalg.norm(question_embedding_mean, ord=2)
    if float(question_embedding_mean_norm.item()) == 0.0:
        return {position: 0.0 for position in context_token_positions}

    scores_by_context_position: Dict[int, float] = {}

    for context_position in context_token_positions:
        context_input_id = int(input_ids[context_position])
        context_embedding = embedding_weight_cpu[context_input_id]
        context_embedding_norm = torch.linalg.norm(context_embedding, ord=2)

        if float(context_embedding_norm.item()) == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = float(
                torch.dot(context_embedding, question_embedding_mean).item()
                / (context_embedding_norm.item() * question_embedding_mean_norm.item())
            )

        scores_by_context_position[context_position] = cosine_similarity

    return scores_by_context_position

# Expand kept context-token positions with a fixed local window radius
def expand_context_positions_with_local_window(
    selected_positions: set[int],
    ordered_context_positions: list[int],
    window_radius: int,
) -> set[int]:
    expanded_positions: set[int] = set()

    if len(ordered_context_positions) == 0:
        return expanded_positions

    context_position_to_rank = {
        context_position: rank
        for rank, context_position in enumerate(ordered_context_positions)
    }

    for selected_position in selected_positions:
        selected_rank = context_position_to_rank[selected_position]

        start_rank = max(0, int(selected_rank) - int(window_radius))
        end_rank = min(len(ordered_context_positions) - 1, int(selected_rank) + int(window_radius))

        for expanded_rank in range(start_rank, end_rank + 1):
            expanded_positions.add(ordered_context_positions[expanded_rank])

    return expanded_positions

# Select context-token positions (with span expansion)
def select_context_positions_to_keep_with_local_window(
    context_token_positions: list[int],
    scores_by_context_position: Dict[int, float],
    keep_ratio: float,
    window_radius: int,
) -> set[int]:
    normalized_keep_ratio = normalize_token_pruning_keep_ratio(keep_ratio)

    if len(context_token_positions) == 0:
        return set()

    target_number_of_context_tokens_to_keep = max(
        1,
        int(math.ceil(normalized_keep_ratio * len(context_token_positions))),
    )

    ranked_context_positions = sorted(
        context_token_positions,
        key=lambda position: (
            scores_by_context_position.get(position, 0.0),
            -position,
        ),
        reverse=True,
    )

    selected_anchor_positions: set[int] = set()
    expanded_positions: set[int] = set()

    for ranked_position in ranked_context_positions:
        selected_anchor_positions.add(ranked_position)

        expanded_positions = expand_context_positions_with_local_window(
            selected_positions=selected_anchor_positions,
            ordered_context_positions=context_token_positions,
            window_radius=int(window_radius),
        )

        if len(expanded_positions) >= target_number_of_context_tokens_to_keep:
            break

    if len(expanded_positions) > target_number_of_context_tokens_to_keep:
        expanded_positions = set(
            sorted(
                expanded_positions,
                key=lambda position: (
                    scores_by_context_position.get(position, 0.0),
                    -position,
                ),
                reverse=True,
            )[:target_number_of_context_tokens_to_keep]
        )

    return expanded_positions

# Prune context tokens while preserving local span structure around selected anchors
def prune_tokenized_feature_with_keep_ratio(
    feature: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    embedding_weight_cpu: torch.Tensor,
    keep_ratio: float,
) -> Dict[str, Any]:
    normalized_keep_ratio = normalize_token_pruning_keep_ratio(keep_ratio)

    input_ids = feature["input_ids"]
    attention_mask = feature["attention_mask"]
    offset_mapping = feature["offset_mapping"]

    question_token_positions = set(
        get_question_token_positions_for_feature(
            feature=feature,
            tokenizer=tokenizer,
        )
    )
    context_token_positions = get_context_token_positions_for_feature(feature=feature)

    if len(context_token_positions) == 0:
        pruned_feature = dict(feature)
        pruned_feature["token_pruning_keep_ratio_target"] = normalized_keep_ratio
        pruned_feature["token_pruning_keep_ratio_realized"] = 1.0
        pruned_feature["number_of_context_tokens_before_pruning"] = 0
        pruned_feature["number_of_context_tokens_after_pruning"] = 0
        return pruned_feature

    scores_by_context_position = score_context_tokens_with_question_embedding_similarity(
        feature=feature,
        tokenizer=tokenizer,
        embedding_weight_cpu=embedding_weight_cpu,
    )

    # Preserve a small local window around high-scoring context-token anchors
    kept_context_positions = select_context_positions_to_keep_with_local_window(
        context_token_positions=context_token_positions,
        scores_by_context_position=scores_by_context_position,
        keep_ratio=normalized_keep_ratio,
        window_radius=2,
    )

    kept_positions: list[int] = []
    special_token_ids = set(tokenizer.all_special_ids)

    for position, (token_id, attention_value, offset_value) in enumerate(
        zip(input_ids, attention_mask, offset_mapping)
    ):
        is_active_token = int(attention_value) == 1
        is_question_token = position in question_token_positions
        is_special_token = is_active_token and int(token_id) in special_token_ids

        should_keep_position = (
            is_question_token
            or is_special_token
            or position in kept_context_positions
        )

        if should_keep_position:
            kept_positions.append(position)

    pruned_feature: Dict[str, Any] = {}

    sequence_length = len(input_ids)
    for field_name, field_value in feature.items():
        if isinstance(field_value, list) and len(field_value) == sequence_length:
            pruned_feature[field_name] = [field_value[position] for position in kept_positions]
        else:
            pruned_feature[field_name] = field_value

    pruned_feature["token_pruning_keep_ratio_target"] = normalized_keep_ratio
    pruned_feature["token_pruning_keep_ratio_realized"] = (
        float(len(kept_context_positions)) / float(len(context_token_positions))
        if len(context_token_positions) > 0
        else 1.0
    )
    pruned_feature["number_of_context_tokens_before_pruning"] = int(len(context_token_positions))
    pruned_feature["number_of_context_tokens_after_pruning"] = int(len(kept_context_positions))

    return pruned_feature

# Apply dynamic question-aware token pruning
def apply_dynamic_token_pruning_to_tokenized_features(
    tokenized_evaluation_features,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    keep_ratio: float,
):
    normalized_keep_ratio = normalize_token_pruning_keep_ratio(keep_ratio)

    embedding_weight_cpu = (
        model.get_input_embeddings().weight.detach().to(device="cpu", dtype=torch.float32)
    )

    pruned_feature_rows = []
    for feature in tokenized_evaluation_features:
        pruned_feature_rows.append(
            prune_tokenized_feature_with_keep_ratio(
                feature=feature,
                tokenizer=tokenizer,
                embedding_weight_cpu=embedding_weight_cpu,
                keep_ratio=normalized_keep_ratio,
            )
        )

    return Dataset.from_list(pruned_feature_rows)

# Evaluate one checkpoint on SQuAD v2 from already-tokenized features
def evaluate_checkpoint_on_squad_v2_with_tokenized_features(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    tokenized_evaluation_features,
    maximum_sequence_length: int,
    effective_document_stride: int,
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    pad_to_multiple_of: Optional[int],
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
    run_label: str,
    extra_result_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features

    tokenized_evaluation_features_for_model = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    evaluation_dataloader = build_evaluation_dataloader(
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        tokenizer=tokenizer,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
    )

    append_line_to_text_file(
        log_file_path,
        f"[{run_label}] running warmup batches={number_of_warmup_batches}",
    )
    run_inference_warmup(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
        number_of_warmup_batches=int(number_of_warmup_batches),
    )

    append_line_to_text_file(
        log_file_path,
        f"[{run_label}] starting inference energy meter",
    )
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    inference_energy_meter.start()

    measured_inference_output = run_measured_question_answering_inference_with_trainer(
        run_directory=run_directory,
        model=model,
        tokenizer=tokenizer,
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
        device=device,
    )

    inference_energy_meter.stop()
    append_line_to_text_file(
        log_file_path,
        f"[{run_label}] stopped inference energy meter",
    )

    start_logits = measured_inference_output["start_logits"]
    end_logits = measured_inference_output["end_logits"]
    inference_runtime_seconds = float(measured_inference_output["inference_runtime_seconds"])
    inference_energy_joules = float(inference_energy_meter.get_energy_joules())
    number_of_energy_samples = int(len(inference_energy_meter.samples))

    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
    )

    thresholded_predictions_by_example_id: Dict[str, str] = {}
    for example_id, prediction_text in predictions_by_example_id.items():
        no_answer_probability = float(no_answer_probability_by_example_id.get(example_id, 0.0))
        thresholded_predictions_by_example_id[example_id] = (
            ""
            if no_answer_probability >= float(no_answer_probability_threshold)
            else prediction_text
        )

    raw_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )
    thresholded_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=thresholded_predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )

    number_of_raw_evaluation_examples = int(len(raw_evaluation_split))
    number_of_feature_windows = int(len(tokenized_evaluation_features_for_model))
    number_of_inference_tokens = int(
        count_non_padding_tokens_in_feature_dataset(tokenized_evaluation_features_for_model)
    )

    average_latency_per_raw_example_milliseconds = (
        (inference_runtime_seconds / number_of_raw_evaluation_examples) * 1000.0
        if number_of_raw_evaluation_examples > 0
        else None
    )
    average_latency_per_feature_window_milliseconds = (
        (inference_runtime_seconds / number_of_feature_windows) * 1000.0
        if number_of_feature_windows > 0
        else None
    )

    thresholded_exact_match_correct_examples = int(
        round((float(thresholded_metrics["exact"]) / 100.0) * number_of_raw_evaluation_examples)
    )
    thresholded_exact_match_correct_examples = max(1, thresholded_exact_match_correct_examples)

    result: Dict[str, Any] = {
        "maximum_sequence_length": int(maximum_sequence_length),
        "effective_document_stride": int(effective_document_stride),
        "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
        "number_of_feature_windows": number_of_feature_windows,
        "number_of_inference_tokens": number_of_inference_tokens,
        "inference_runtime_seconds": inference_runtime_seconds,
        "average_latency_per_raw_example_milliseconds": average_latency_per_raw_example_milliseconds,
        "average_latency_per_feature_window_milliseconds": average_latency_per_feature_window_milliseconds,
        "inference_energy_joules": inference_energy_joules,
        "number_of_energy_samples": number_of_energy_samples,
        "joules_per_inference_example": (
            inference_energy_joules / number_of_raw_evaluation_examples
            if number_of_raw_evaluation_examples > 0
            else None
        ),
        "joules_per_feature_window": (
            inference_energy_joules / number_of_feature_windows
            if number_of_feature_windows > 0
            else None
        ),
        "joules_per_inference_token": (
            inference_energy_joules / number_of_inference_tokens
            if number_of_inference_tokens > 0
            else None
        ),
        "joules_per_exact_match_correct_example": (
            inference_energy_joules / thresholded_exact_match_correct_examples
            if thresholded_exact_match_correct_examples > 0
            else None
        ),
        "no_answer_probability_threshold": float(no_answer_probability_threshold),
        "metrics_raw": raw_metrics,
        "metrics_thresholded": thresholded_metrics,
    }

    if extra_result_fields is not None:
        result.update(extra_result_fields)

    write_json_file(result, run_directory / "result.json")

    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields={
            "phase": "inference",
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
            "number_of_feature_windows": number_of_feature_windows,
            "number_of_inference_tokens": number_of_inference_tokens,
            "joules_per_inference_example": result["joules_per_inference_example"],
            "joules_per_feature_window": result["joules_per_feature_window"],
            "joules_per_inference_token": result["joules_per_inference_token"],
            "number_of_energy_samples": number_of_energy_samples,
            **(extra_result_fields or {}),
        },
    )

    return result

# Evaluate one checkpoint on SQuAD v2 at specific inference max sequence length
def evaluate_checkpoint_on_squad_v2_at_sequence_length(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
    pad_to_multiple_of: Optional[int],
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    # Clamp document stride for shorter sequence lengths
    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] effective_document_stride={effective_document_stride}",
    )

    # Tokenize evaluation features for this specific sequence length
    tokenized_evaluation_features = raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(effective_document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc=f"Tokenizing SQuAD v2 eval at max_sequence_length={maximum_sequence_length}",
    )

    # Keep full feature set for postprocessing
    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features

    # Remove non-tensor columns for model batching
    tokenized_evaluation_features_for_model = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    # Build the measured dataloader
    evaluation_dataloader = build_evaluation_dataloader(
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        tokenizer=tokenizer,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Run short warmup pass
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] running warmup batches={number_of_warmup_batches}",
    )
    run_inference_warmup(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
        number_of_warmup_batches=int(number_of_warmup_batches),
    )

    # Measure energy during full evaluation pass
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] starting inference energy meter",
    )
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    inference_energy_meter.start()

    measured_inference_output = run_measured_question_answering_inference_with_trainer(
        run_directory=run_directory,
        model=model,
        tokenizer=tokenizer,
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
        device=device,
    )

    inference_energy_meter.stop()
    append_line_to_text_file(
        log_file_path,
        f"[sequence_length={maximum_sequence_length}] stopped inference energy meter",
    )

    start_logits = measured_inference_output["start_logits"]
    end_logits = measured_inference_output["end_logits"]
    inference_runtime_seconds = float(measured_inference_output["inference_runtime_seconds"])
    inference_energy_joules = float(inference_energy_meter.get_energy_joules())
    number_of_energy_samples = int(len(inference_energy_meter.samples))

    # Postprocess logits into text predictions and no-answer probabilities
    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
    )

    # Apply fixed threshold to emit explicit empty-string predictions
    thresholded_predictions_by_example_id: Dict[str, str] = {}
    for example_id, prediction_text in predictions_by_example_id.items():
        no_answer_probability = float(no_answer_probability_by_example_id.get(example_id, 0.0))
        thresholded_predictions_by_example_id[example_id] = (
            ""
            if no_answer_probability >= float(no_answer_probability_threshold)
            else prediction_text
        )

    # Compute raw and thresholded SQuAD v2 metrics
    raw_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )
    thresholded_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=thresholded_predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )

    # Count raw examples, feature windows, and tokens for normalized reporting
    number_of_raw_evaluation_examples = int(len(raw_evaluation_split))
    number_of_feature_windows = int(len(tokenized_evaluation_features_for_model))
    number_of_inference_tokens = int(
        count_non_padding_tokens_in_feature_dataset(tokenized_evaluation_features_for_model)
    )

    average_latency_per_raw_example_milliseconds = (
        (inference_runtime_seconds / number_of_raw_evaluation_examples) * 1000.0
        if number_of_raw_evaluation_examples > 0
        else None
    )
    average_latency_per_feature_window_milliseconds = (
        (inference_runtime_seconds / number_of_feature_windows) * 1000.0
        if number_of_feature_windows > 0
        else None
    )

    thresholded_exact_match_correct_examples = int(
        round((float(thresholded_metrics["exact"]) / 100.0) * number_of_raw_evaluation_examples)
    )
    thresholded_exact_match_correct_examples = max(1, thresholded_exact_match_correct_examples)

    result = {
        "maximum_sequence_length": int(maximum_sequence_length),
        "effective_document_stride": int(effective_document_stride),
        "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
        "number_of_feature_windows": number_of_feature_windows,
        "number_of_inference_tokens": number_of_inference_tokens,
        "inference_runtime_seconds": inference_runtime_seconds,
        "average_latency_per_raw_example_milliseconds": average_latency_per_raw_example_milliseconds,
        "average_latency_per_feature_window_milliseconds": average_latency_per_feature_window_milliseconds,
        "inference_energy_joules": inference_energy_joules,
        "number_of_energy_samples": number_of_energy_samples,
        "joules_per_inference_example": (
            inference_energy_joules / number_of_raw_evaluation_examples
            if number_of_raw_evaluation_examples > 0
            else None
        ),
        "joules_per_feature_window": (
            inference_energy_joules / number_of_feature_windows
            if number_of_feature_windows > 0
            else None
        ),
        "joules_per_inference_token": (
            inference_energy_joules / number_of_inference_tokens
            if number_of_inference_tokens > 0
            else None
        ),
        "joules_per_exact_match_correct_example": (
            inference_energy_joules / thresholded_exact_match_correct_examples
            if thresholded_exact_match_correct_examples > 0
            else None
        ),
        "no_answer_probability_threshold": float(no_answer_probability_threshold),
        "metrics_raw": raw_metrics,
        "metrics_thresholded": thresholded_metrics,
    }

    # Persist full per-length result for debugging and reproducibility
    write_json_file(result, run_directory / "result.json")

    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields={
            "phase": "inference",
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
            "number_of_feature_windows": number_of_feature_windows,
            "number_of_inference_tokens": number_of_inference_tokens,
            "joules_per_inference_example": result["joules_per_inference_example"],
            "joules_per_feature_window": result["joules_per_feature_window"],
            "joules_per_inference_token": result["joules_per_inference_token"],
            "number_of_energy_samples": number_of_energy_samples,
        },
    )

    return result

# Tokenize SQuAD v2 evaluation features once for reuse across token-pruning sweep values
def tokenize_squad_v2_evaluation_features_once(
    raw_evaluation_split,
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
):
    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    tokenized_evaluation_features = raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(effective_document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc=f"Tokenizing SQuAD v2 eval at max_sequence_length={maximum_sequence_length}",
    )

    return tokenized_evaluation_features, int(effective_document_stride)

# Evaluate one checkpoint on SQuAD v2 at specific inference precision
def evaluate_checkpoint_on_squad_v2_at_precision(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
    pad_to_multiple_of: Optional[int],
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
    precision_mode: str,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    normalized_precision_mode = normalize_precision_mode_name(precision_mode)

    precision_supported, precision_skip_reason = check_precision_mode_support(
        precision_mode=normalized_precision_mode,
        device=device,
    )

    effective_document_stride = clamp_document_stride_for_sequence_length(
        configured_document_stride=configured_document_stride,
        maximum_sequence_length=maximum_sequence_length,
    )

    append_line_to_text_file(
        log_file_path,
        f"[precision={normalized_precision_mode}] effective_document_stride={effective_document_stride}",
    )

    append_line_to_text_file(
        log_file_path,
        f"[precision={normalized_precision_mode}] precision_supported={precision_supported}",
    )

    if not precision_supported:
        result = {
            "precision_mode": normalized_precision_mode,
            "precision_supported": False,
            "precision_skip_reason": precision_skip_reason,
            "model_parameter_dtype": None,
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": int(len(raw_evaluation_split)),
            "number_of_feature_windows": None,
            "number_of_inference_tokens": None,
            "inference_runtime_seconds": None,
            "average_latency_per_raw_example_milliseconds": None,
            "average_latency_per_feature_window_milliseconds": None,
            "inference_energy_joules": None,
            "number_of_energy_samples": 0,
            "joules_per_inference_example": None,
            "joules_per_feature_window": None,
            "joules_per_inference_token": None,
            "joules_per_exact_match_correct_example": None,
            "no_answer_probability_threshold": float(no_answer_probability_threshold),
            "metrics_raw": None,
            "metrics_thresholded": None,
        }

        write_json_file(result, run_directory / "result.json")
        return result

    model = cast_model_for_inference_precision(
        model=model,
        precision_mode=normalized_precision_mode,
    )
    model.eval()

    tokenized_evaluation_features = raw_evaluation_split.map(
        lambda examples: prepare_squad_v2_evaluation_features(
            examples=examples,
            tokenizer=tokenizer,
            maximum_sequence_length=int(maximum_sequence_length),
            document_stride=int(effective_document_stride),
            pad_to_maximum_length=bool(pad_to_maximum_length),
        ),
        batched=True,
        remove_columns=raw_evaluation_split.column_names,
        desc=(
            f"Tokenizing SQuAD v2 eval at max_sequence_length={maximum_sequence_length} "
            f"for precision={normalized_precision_mode}"
        ),
    )

    tokenized_evaluation_features_for_postprocessing = tokenized_evaluation_features

    tokenized_evaluation_features_for_model = tokenized_evaluation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    evaluation_dataloader = build_evaluation_dataloader(
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        tokenizer=tokenizer,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
    )

    append_line_to_text_file(
        log_file_path,
        f"[precision={normalized_precision_mode}] running warmup batches={number_of_warmup_batches}",
    )
    run_inference_warmup(
        model=model,
        dataloader=evaluation_dataloader,
        device=device,
        number_of_warmup_batches=int(number_of_warmup_batches),
    )

    append_line_to_text_file(
        log_file_path,
        f"[precision={normalized_precision_mode}] starting inference energy meter",
    )
    inference_energy_meter = EnergyMeter(
        sampling_interval_seconds=float(power_sampling_interval_seconds)
    )
    inference_energy_meter.start()

    measured_inference_output = run_measured_question_answering_inference_with_trainer(
        run_directory=run_directory,
        model=model,
        tokenizer=tokenizer,
        tokenized_evaluation_features_for_model=tokenized_evaluation_features_for_model,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
        device=device,
    )

    inference_energy_meter.stop()
    append_line_to_text_file(
        log_file_path,
        f"[precision={normalized_precision_mode}] stopped inference energy meter",
    )

    start_logits = measured_inference_output["start_logits"]
    end_logits = measured_inference_output["end_logits"]
    inference_runtime_seconds = float(measured_inference_output["inference_runtime_seconds"])
    inference_energy_joules = float(inference_energy_meter.get_energy_joules())
    number_of_energy_samples = int(len(inference_energy_meter.samples))

    predictions_by_example_id, no_answer_probability_by_example_id = postprocess_squad_v2_predictions(
        raw_examples=raw_evaluation_split,
        tokenized_features=tokenized_evaluation_features_for_postprocessing,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
    )

    thresholded_predictions_by_example_id: Dict[str, str] = {}
    for example_id, prediction_text in predictions_by_example_id.items():
        no_answer_probability = float(no_answer_probability_by_example_id.get(example_id, 0.0))
        thresholded_predictions_by_example_id[example_id] = (
            ""
            if no_answer_probability >= float(no_answer_probability_threshold)
            else prediction_text
        )

    raw_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )
    thresholded_metrics = compute_squad_v2_metrics(
        predictions_by_example_id=thresholded_predictions_by_example_id,
        no_answer_probability_by_example_id=no_answer_probability_by_example_id,
        raw_evaluation_dataset=raw_evaluation_split,
    )

    number_of_raw_evaluation_examples = int(len(raw_evaluation_split))
    number_of_feature_windows = int(len(tokenized_evaluation_features_for_model))
    number_of_inference_tokens = int(
        count_non_padding_tokens_in_feature_dataset(tokenized_evaluation_features_for_model)
    )

    average_latency_per_raw_example_milliseconds = (
        (inference_runtime_seconds / number_of_raw_evaluation_examples) * 1000.0
        if number_of_raw_evaluation_examples > 0
        else None
    )
    average_latency_per_feature_window_milliseconds = (
        (inference_runtime_seconds / number_of_feature_windows) * 1000.0
        if number_of_feature_windows > 0
        else None
    )

    thresholded_exact_match_correct_examples = int(
        round((float(thresholded_metrics["exact"]) / 100.0) * number_of_raw_evaluation_examples)
    )
    thresholded_exact_match_correct_examples = max(1, thresholded_exact_match_correct_examples)

    result: Dict[str, Any] = {
        "precision_mode": normalized_precision_mode,
        "precision_supported": True,
        "precision_skip_reason": None,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "maximum_sequence_length": int(maximum_sequence_length),
        "effective_document_stride": int(effective_document_stride),
        "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
        "number_of_feature_windows": number_of_feature_windows,
        "number_of_inference_tokens": number_of_inference_tokens,
        "inference_runtime_seconds": inference_runtime_seconds,
        "average_latency_per_raw_example_milliseconds": average_latency_per_raw_example_milliseconds,
        "average_latency_per_feature_window_milliseconds": average_latency_per_feature_window_milliseconds,
        "inference_energy_joules": inference_energy_joules,
        "number_of_energy_samples": number_of_energy_samples,
        "joules_per_inference_example": (
            inference_energy_joules / number_of_raw_evaluation_examples
            if number_of_raw_evaluation_examples > 0
            else None
        ),
        "joules_per_feature_window": (
            inference_energy_joules / number_of_feature_windows
            if number_of_feature_windows > 0
            else None
        ),
        "joules_per_inference_token": (
            inference_energy_joules / number_of_inference_tokens
            if number_of_inference_tokens > 0
            else None
        ),
        "joules_per_exact_match_correct_example": (
            inference_energy_joules / thresholded_exact_match_correct_examples
            if thresholded_exact_match_correct_examples > 0
            else None
        ),
        "no_answer_probability_threshold": float(no_answer_probability_threshold),
        "metrics_raw": raw_metrics,
        "metrics_thresholded": thresholded_metrics,
    }

    write_json_file(result, run_directory / "result.json")

    inference_energy_meter.save_report(
        path=run_directory / "energy_infer.json",
        additional_fields={
            "phase": "inference",
            "precision_mode": normalized_precision_mode,
            "maximum_sequence_length": int(maximum_sequence_length),
            "effective_document_stride": int(effective_document_stride),
            "number_of_raw_evaluation_examples": number_of_raw_evaluation_examples,
            "number_of_feature_windows": number_of_feature_windows,
            "number_of_inference_tokens": number_of_inference_tokens,
            "joules_per_inference_example": result["joules_per_inference_example"],
            "joules_per_feature_window": result["joules_per_feature_window"],
            "joules_per_inference_token": result["joules_per_inference_token"],
            "number_of_energy_samples": number_of_energy_samples,
        },
    )

    return result

# Evaluate one checkpoint on SQuAD v2 at specific dynamic token-pruning keep ratio
def evaluate_checkpoint_on_squad_v2_with_token_pruning(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    maximum_sequence_length: int,
    configured_document_stride: int,
    pad_to_maximum_length: bool,
    pad_to_multiple_of: Optional[int],
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
    token_pruning_keep_ratio: float,
) -> Dict[str, Any]:
    tokenized_evaluation_features, effective_document_stride = tokenize_squad_v2_evaluation_features_once(
        raw_evaluation_split=raw_evaluation_split,
        tokenizer=tokenizer,
        maximum_sequence_length=int(maximum_sequence_length),
        configured_document_stride=int(configured_document_stride),
        pad_to_maximum_length=bool(pad_to_maximum_length),
    )

    return evaluate_checkpoint_on_squad_v2_with_token_pruning_from_tokenized_features(
        run_directory=run_directory,
        log_file_path=log_file_path,
        model=model,
        tokenizer=tokenizer,
        raw_evaluation_split=raw_evaluation_split,
        tokenized_evaluation_features=tokenized_evaluation_features,
        maximum_sequence_length=int(maximum_sequence_length),
        effective_document_stride=int(effective_document_stride),
        pad_to_multiple_of=pad_to_multiple_of,
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        number_of_warmup_batches=int(number_of_warmup_batches),
        power_sampling_interval_seconds=float(power_sampling_interval_seconds),
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
        no_answer_probability_threshold=float(no_answer_probability_threshold),
        token_pruning_keep_ratio=float(token_pruning_keep_ratio),
    )

# Evaluate one checkpoint on SQuAD v2 with token pruning from pre-tokenized reusable features
def evaluate_checkpoint_on_squad_v2_with_token_pruning_from_tokenized_features(
    run_directory: Path,
    log_file_path: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    raw_evaluation_split,
    tokenized_evaluation_features,
    maximum_sequence_length: int,
    effective_document_stride: int,
    pad_to_multiple_of: Optional[int],
    per_device_evaluation_batch_size: int,
    dataloader_num_workers: int,
    number_of_warmup_batches: int,
    power_sampling_interval_seconds: float,
    n_best_size: int,
    maximum_answer_length: int,
    no_answer_probability_threshold: float,
    token_pruning_keep_ratio: float,
) -> Dict[str, Any]:
    normalized_keep_ratio = normalize_token_pruning_keep_ratio(token_pruning_keep_ratio)
    keep_ratio_label = format_token_pruning_keep_ratio_label(normalized_keep_ratio)

    append_line_to_text_file(
        log_file_path,
        f"[token_pruning_keep_ratio={keep_ratio_label}] effective_document_stride={effective_document_stride}",
    )

    append_line_to_text_file(
        log_file_path,
        f"[token_pruning_keep_ratio={keep_ratio_label}] applying dynamic token pruning",
    )

    pruned_tokenized_evaluation_features = apply_dynamic_token_pruning_to_tokenized_features(
        tokenized_evaluation_features=tokenized_evaluation_features,
        tokenizer=tokenizer,
        model=model,
        keep_ratio=normalized_keep_ratio,
    )

    number_of_context_tokens_before_pruning = int(
        sum(
            int(feature["number_of_context_tokens_before_pruning"])
            for feature in pruned_tokenized_evaluation_features
        )
    )
    number_of_context_tokens_after_pruning = int(
        sum(
            int(feature["number_of_context_tokens_after_pruning"])
            for feature in pruned_tokenized_evaluation_features
        )
    )

    realized_context_keep_ratio = (
        float(number_of_context_tokens_after_pruning) / float(number_of_context_tokens_before_pruning)
        if number_of_context_tokens_before_pruning > 0
        else 1.0
    )

    pruned_tokenized_evaluation_features = pruned_tokenized_evaluation_features.remove_columns(
        [
            "token_pruning_keep_ratio_target",
            "token_pruning_keep_ratio_realized",
            "number_of_context_tokens_before_pruning",
            "number_of_context_tokens_after_pruning",
        ]
    )

    return evaluate_checkpoint_on_squad_v2_with_tokenized_features(
        run_directory=run_directory,
        log_file_path=log_file_path,
        model=model,
        tokenizer=tokenizer,
        raw_evaluation_split=raw_evaluation_split,
        tokenized_evaluation_features=pruned_tokenized_evaluation_features,
        maximum_sequence_length=int(maximum_sequence_length),
        effective_document_stride=int(effective_document_stride),
        per_device_evaluation_batch_size=int(per_device_evaluation_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        pad_to_multiple_of=pad_to_multiple_of,
        number_of_warmup_batches=int(number_of_warmup_batches),
        power_sampling_interval_seconds=float(power_sampling_interval_seconds),
        n_best_size=int(n_best_size),
        maximum_answer_length=int(maximum_answer_length),
        no_answer_probability_threshold=float(no_answer_probability_threshold),
        run_label=f"token_pruning_keep_ratio={keep_ratio_label}",
        extra_result_fields={
            "token_pruning_keep_ratio": normalized_keep_ratio,
            "token_pruning_keep_ratio_label": keep_ratio_label,
            "number_of_context_tokens_before_pruning": number_of_context_tokens_before_pruning,
            "number_of_context_tokens_after_pruning": number_of_context_tokens_after_pruning,
            "realized_context_keep_ratio": realized_context_keep_ratio,
        },
    )