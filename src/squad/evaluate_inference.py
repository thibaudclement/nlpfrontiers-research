import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering
import contextlib
import evaluate
from .measure import GPUPowerSampler, integrate_energy, save_power_trace_csv


# Bundle inference benchmark outputs for logging
@dataclass
class InferenceBenchmarkResult:
    exact_match: float
    f1: float
    average_latency_per_example_ms: float
    energy_per_example_j: float
    energy_per_correct_j: float
    peak_gpu_memory_mb: float


# Convert start/end logits over features into one prediction per example
def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int,
    max_answer_length: int,
) -> Dict[str, str]:
    all_start_logits, all_end_logits = raw_predictions

    # Map example id to the list of feature indices
    features_per_example: Dict[str, List[int]] = {}
    for feature_index, feature in enumerate(features):
        example_id = feature["example_id"]
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(feature_index)

    predictions: Dict[str, str] = {}

    # Generate one prediction per example by searching best spans across all its features
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        feature_indices = features_per_example.get(example_id, [])

        valid_answers: List[Dict[str, object]] = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # Select top-N start and end indices by logit score
            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1].tolist()
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid indices
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue

                    # Skip non-context tokens (stored as None)
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue

                    # Skip invalid spans
                    if end_index < start_index:
                        continue

                    # Skip overly long spans
                    span_length = end_index - start_index + 1
                    if span_length > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    text = context[start_char:end_char]

                    score = float(start_logits[start_index] + end_logits[end_index])
                    valid_answers.append({"score": score, "text": text})

        # Select best answer text (fallback to empty string)
        if len(valid_answers) == 0:
            predictions[example_id] = ""
        else:
            best = max(valid_answers, key = lambda x: x["score"])
            predictions[example_id] = best["text"]

    return predictions


# Build an inference autocast context for specified precision
def get_inference_autocast_context(precision: str, device_type: str):
    precision = precision.lower().strip()

    # No autocast necessary for CPU
    if device_type != "cuda":
        return contextlib.nullcontext()

    # No autocast necessary for full precision (fp32)
    if precision == "fp32":
        return contextlib.nullcontext()

    # Apply autocast for half precision (fp16)
    if precision == "fp16":
        return torch.autocast(device_type, dtype = torch.float16)

    # Attempt to apply autocast for quarter precision (fp8) if supported
    if precision == "fp8":
        try:
            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is not None:
                return torch.autocast(device_type="cuda", dtype = fp8_dtype)
        except Exception:
            pass

        try:
            import transformer_engine.pytorch as te  # type: ignore
            return te.fp8_autocast(enabled = True)
        except Exception as e:
            raise RuntimeError(
                "FP8 requested but neither torch FP8 autocast nor transformer-engine fp8_autocast is available."
            ) from e

    raise ValueError(f"Unsupported precision '{precision}'. Expected fp32, fp16, or fp8.")


# Create a tensor-only dataset view for DataLoader collation
def _make_tensor_only_features(validation_features):
    # Remove columns that contain non-tensor values (None in offset_mapping, and string ids)
    columns_to_remove = []
    for col in ["offset_mapping", "example_id"]:
        if col in validation_features.column_names:
            columns_to_remove.append(col)

    if len(columns_to_remove) == 0:
        return validation_features

    return validation_features.remove_columns(columns_to_remove)


# Run QA inference benchmark and compute EM/F1 plus energy/latency
def benchmark_inference_qa(
    run_directory: Path,
    model_directory: str,
    validation_features: object,
    validation_examples: object,
    evaluation_batch_size: int,
    num_inference_batches: int,
    power_sample_interval_s: float,
    n_best_size: int,
    max_answer_length: int,
    gpu_index: int = 0,
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load QA model
    model = AutoModelForQuestionAnswering.from_pretrained(model_directory)
    model.to(device)
    model.eval()

    # Create a tensor-only view of features for batching (avoids NoneType collation)
    model_features = _make_tensor_only_features(validation_features)

    # Build DataLoader over model-only features
    loader = DataLoader(model_features, batch_size=evaluation_batch_size, shuffle=False)

    # Reset peak memory stats before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm up to stabilize kernels/caches
    warmup_batches = 10
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup_batches:
                break

            # Keep only fields expected by the QA model
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            _ = model(**batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Start GPU power sampling
    sampler = GPUPowerSampler(gpu_index = gpu_index, sample_interval_s = power_sample_interval_s)
    sampler.start()

    start_logits_list: List[np.ndarray] = []
    end_logits_list: List[np.ndarray] = []
    start_time = time.perf_counter()

    measured_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}

            if device.type == "cuda":
                torch.cuda.synchronize()

            outputs = model(**batch)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_logits_list.append(outputs.start_logits.detach().cpu().numpy())
            end_logits_list.append(outputs.end_logits.detach().cpu().numpy())

            measured_batches += 1
            if measured_batches >= num_inference_batches:
                break

    end_time = time.perf_counter()
    sampler.stop()
    samples = sampler.get_samples()

    # Save power trace
    save_power_trace_csv(samples, run_directory / "power_trace.csv")

    # Concatenate logits across processed features
    all_start_logits = np.concatenate(start_logits_list, axis = 0)
    all_end_logits = np.concatenate(end_logits_list, axis = 0)

    # Slice the *original* validation_features (with offset_mapping/example_id) to match processed feature count
    num_features = all_start_logits.shape[0]
    processed_features = validation_features.select(range(num_features))

    # Postprocess predictions at example level
    predictions_by_id = postprocess_qa_predictions(
        examples = validation_examples,
        features = processed_features,
        raw_predictions = (all_start_logits, all_end_logits),
        n_best_size = n_best_size,
        max_answer_length = max_answer_length,
    )

    # Build metric inputs
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions_by_id.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in validation_examples]

    # Compute SQuAD EM/F1
    squad_metric = evaluate.load("squad")
    metric_output = squad_metric.compute(predictions = formatted_predictions, references = references)
    exact_match = float(metric_output["exact_match"])
    f1 = float(metric_output["f1"])

    # Compute latency and energy normalized per example (not per feature)
    total_examples = len(validation_examples)
    total_time_s = end_time - start_time
    average_latency_per_example_ms = (total_time_s / total_examples) * 1000.0

    total_energy_j = integrate_energy(samples)
    energy_per_example_j = total_energy_j / total_examples

    # Define "correct" as exact match for energy_per_correct
    num_correct = max(1, int(round(exact_match / 100.0 * total_examples)))
    energy_per_correct_j = total_energy_j / num_correct

    # Read peak memory usage
    peak_gpu_memory_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    # Save predictions for quick inspection
    with open(run_directory / "predictions.jsonl", "w", encoding = "utf-8") as f:
        for pred in formatted_predictions[:200]:
            f.write(json.dumps(pred) + "\n")

    return InferenceBenchmarkResult(
        exact_match = exact_match,
        f1 = f1,
        average_latency_per_example_ms = average_latency_per_example_ms,
        energy_per_example_j = energy_per_example_j,
        energy_per_correct_j = energy_per_correct_j,
        peak_gpu_memory_mb = peak_gpu_memory_mb,
    )


# Run QA inference benchmark with specified precision and compute EM/F1 plus energy/latency
def benchmark_inference_qa_with_precision(
    run_directory: Path,
    model_directory: str,
    precision: str,
    validation_features: object,
    validation_examples: object,
    evaluation_batch_size: int,
    num_inference_batches: int,
    power_sample_interval_s: float,
    n_best_size: int,
    max_answer_length: int,
    gpu_index: int = 0,
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load QA model
    model = AutoModelForQuestionAnswering.from_pretrained(model_directory)
    model.to(device)
    model.eval()

    # Create a tensor-only view of features for batching (avoids NoneType collation)
    model_features = _make_tensor_only_features(validation_features)

    # Build DataLoader over model-only features
    loader = DataLoader(model_features, batch_size = evaluation_batch_size, shuffle = False)

    # Reset peak memory stats before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    autocast_context = get_inference_autocast_context(precision, device.type)

    # Warm up to stabilize kernels/caches
    warmup_batches = 10
    with torch.no_grad():
        with autocast_context:
            for i, batch in enumerate(loader):
                if i >= warmup_batches:
                    break
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
                _ = model(**batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Start GPU power sampling
    sampler = GPUPowerSampler(gpu_index = gpu_index, sample_interval_s = power_sample_interval_s)
    sampler.start()

    start_logits_list: List[np.ndarray] = []
    end_logits_list: List[np.ndarray] = []
    start_time = time.perf_counter()

    measured_batches = 0
    with torch.no_grad():
        with autocast_context:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}

                if device.type == "cuda":
                    torch.cuda.synchronize()

                outputs = model(**batch)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                start_logits_list.append(outputs.start_logits.detach().cpu().numpy())
                end_logits_list.append(outputs.end_logits.detach().cpu().numpy())

                measured_batches += 1
                if measured_batches >= num_inference_batches:
                    break

    end_time = time.perf_counter()
    sampler.stop()
    samples = sampler.get_samples()

    # Save power trace
    save_power_trace_csv(samples, run_directory / "power_trace.csv")

    # Concatenate logits across processed features
    all_start_logits = np.concatenate(start_logits_list, axis = 0)
    all_end_logits = np.concatenate(end_logits_list, axis = 0)

    # Slice the *original* validation_features (with offset_mapping/example_id) to match processed feature count
    num_features = all_start_logits.shape[0]
    processed_features = validation_features.select(range(num_features))

    # Postprocess predictions at example level
    predictions_by_id = postprocess_qa_predictions(
        examples = validation_examples,
        features = processed_features,
        raw_predictions = (all_start_logits, all_end_logits),
        n_best_size = n_best_size,
        max_answer_length = max_answer_length,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions_by_id.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in validation_examples]

    squad_metric = evaluate.load("squad")
    metric_output = squad_metric.compute(predictions = formatted_predictions, references = references)
    exact_match = float(metric_output["exact_match"])
    f1 = float(metric_output["f1"])

    total_examples = len(validation_examples)
    total_time_s = end_time - start_time
    average_latency_per_example_ms = (total_time_s / total_examples) * 1000.0

    total_energy_j = integrate_energy(samples)
    energy_per_example_j = total_energy_j / total_examples

    num_correct = max(1, int(round(exact_match / 100.0 * total_examples)))
    energy_per_correct_j = total_energy_j / num_correct

    peak_gpu_memory_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    with open(run_directory / "predictions.jsonl", "w", encoding = "utf-8") as f:
        for pred in formatted_predictions[:200]:
            f.write(json.dumps(pred) + "\n")

    return InferenceBenchmarkResult(
        exact_match = exact_match,
        f1 = f1,
        average_latency_per_example_ms = average_latency_per_example_ms,
        energy_per_example_j = energy_per_example_j,
        energy_per_correct_j = energy_per_correct_j,
        peak_gpu_memory_mb = peak_gpu_memory_mb,
    )