import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from .measure import GPUPowerSampler, integrate_energy, save_power_trace_csv
import contextlib

# Bundle inference benchmark outputs for logging
@dataclass
class InferenceBenchmarkResult:
    accuracy: float
    average_latency_per_example_ms: float
    energy_per_example_j: float
    energy_per_correct_j: float
    peak_gpu_memory_mb: float

# Run inference benchmark
def benchmark_inference(
    run_directory: Path,
    model_directory: str,
    evaluation_dataset: object,
    evaluation_batch_size: int,
    num_inference_batches: int,
    power_sample_interval_s: float,
    gpu_index: int = 0
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    model.to(device)
    model.eval()

    loader = DataLoader(evaluation_dataset, batch_size = evaluation_batch_size, shuffle = False)

    # Reset peak memory stats before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm up to stabilize kernels/caches
    warmup_batches = 10
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Start GPU power sampling
    sampler = GPUPowerSampler(gpu_index = gpu_index, sample_interval_s = power_sample_interval_s)
    sampler.start()

    all_predictions = []
    all_labels = []
    start_time = time.perf_counter()

    # Measure fixed number of batches for comparison
    measured_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
        
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            logits = model(**batch).logits

            if device.type == "cuda":
                torch.cuda.synchronize()
            
            predictions = torch.argmax(logits, dim = -1).cpu().numpy().tolist()
            labels = batch["labels"].detach().cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            measured_batches += 1
            if measured_batches >= num_inference_batches:
                break
    
    end_time = time.perf_counter()
    sampler.stop()
    samples = sampler.get_samples()

    # Save power trace
    save_power_trace_csv(samples, run_directory / "power_trace.csv")

    # Compute accuracy
    accuracy = float(np.mean(np.array(all_predictions) == np.array(all_labels)))

    # Compute latency per example
    total_examples = len(all_labels)
    total_time_s = end_time - start_time
    average_latency_per_example_ms = (total_time_s / total_examples) * 1000.0

    # Compute energy per example and per correct prediction
    total_energy_j = integrate_energy(samples)
    energy_per_example_j = total_energy_j / total_examples
    num_correct = int(np.sum(np.array(all_predictions) == np.array(all_labels)))
    energy_per_correct_j = total_energy_j / max(1, num_correct)

    # Read peak memory
    peak_gpu_memory_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    # Save predictions for analysis
    with open(run_directory / "predictions.jsonl", "w", encoding = "utf-8") as f:
        for prediction, label in zip(all_predictions, all_labels):
            f.write(json.dumps({"prediction": int(prediction), "label": int(label)}) + "\n")
    
    return InferenceBenchmarkResult(
        accuracy = accuracy,
        average_latency_per_example_ms = average_latency_per_example_ms,
        energy_per_example_j = energy_per_example_j,
        energy_per_correct_j = energy_per_correct_j,
        peak_gpu_memory_mb = peak_gpu_memory_mb,
    )

# Run inference benchmark with pre-loaded model
def benchmark_inference_model(
        run_directory: Path,
        model: torch.nn.Module,
        evaluation_dataset: object,
        evaluation_batch_size: int,
        num_inference_batches: int,
        power_sample_interval_s: float,
        gpu_index: int = 0
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(evaluation_dataset, batch_size = evaluation_batch_size, shuffle = False)

    # Reset peak memory stats before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm up to stabilize kernels/caches
    warmup_batches = 10
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Start GPU power sampling
    sampler = GPUPowerSampler(gpu_index = gpu_index, sample_interval_s = power_sample_interval_s)
    sampler.start()

    all_predictions: List[int] = []
    all_labels: List[int] = []
    start_time = time.perf_counter()

    # Measure fixed number of batches for comparison
    measured_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if device.type == "cuda":
                torch.cuda.synchronize()

            logits = model(**batch).logits

            if device.type == "cuda":
                torch.cuda.synchronize()

            predictions = torch.argmax(logits, dim = -1).detach().cpu().numpy().tolist()
            labels = batch["labels"].detach().cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            measured_batches += 1
            if measured_batches >= num_inference_batches:
                break

    end_time = time.perf_counter()
    sampler.stop()
    samples = sampler.get_samples()

    # Save power trace
    save_power_trace_csv(samples, run_directory / "power_trace.csv")

    # Compute accuracy
    accuracy = float(np.mean(np.array(all_predictions) == np.array(all_labels)))

    # Compute latency per example
    total_examples = len(all_labels)
    total_time_s = end_time - start_time
    average_latency_per_example_ms = (total_time_s / total_examples) * 1000.0

    # Compute energy per example and per correct prediction
    total_energy_j = integrate_energy(samples)
    energy_per_example_j = total_energy_j / total_examples
    num_correct = int(np.sum(np.array(all_predictions) == np.array(all_labels)))
    energy_per_correct_j = total_energy_j / max(1, num_correct)

    # Read peak memory
    peak_gpu_memory_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    
    # Save predictions for analysis
    with open(run_directory / "predictions.jsonl", "w", encoding = "utf-8") as f:
        for prediction, label in zip(all_predictions, all_labels):
            f.write(json.dumps({"prediction": int(prediction), "label": int(label)}) + "\n")
    
    return InferenceBenchmarkResult(
        accuracy = accuracy,
        average_latency_per_example_ms = average_latency_per_example_ms,
        energy_per_example_j = energy_per_example_j,
        energy_per_correct_j = energy_per_correct_j,
        peak_gpu_memory_mb = peak_gpu_memory_mb,
    )

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
        # Note: PyTorch FP8 autocast is only supported on certain builds
        try:
            # Note: Whether torch.float8_e4m3fn or torch.float8_e5m2 is supported depends on build
            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is not None:
                return torch.autocast(device_type = "cuda", dtype = fp8_dtype)
        except Exception:
            pass

        # Attempt to use NVIDIA Transformer Engine  fp8 autocast if available
        try:
            import transformer_engine.pytorch as te
            return te.fp8_autocast(enabled = True)
        except Exception as e:
            raise RuntimeError(
                "FP8 requested but neither torch FP8 autocast nor transformer-engine fp8_autocast is available."
            ) from e
        
    raise ValueError(f"Unsupported precision '{precision}'. Expected fp32, fp16, bf16, or fp8.")

# Run inference benchmark with with pre-loaded model and specified precision autocast context
def benchmark_inference_model_with_precision(
        run_directory: Path,
        model: torch.nn.Module,
        precision: str,
        evaluation_dataset: object,
        evaluation_batch_size: int,
        num_inference_batches: int,
        power_sample_interval_s: float,
        gpu_index: int = 0
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(evaluation_dataset, batch_size = evaluation_batch_size, shuffle = False)

    # Reset peak memory stats before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm up to stabilize kernels/caches
    autocast_context = get_inference_autocast_context(precision, device.type)
    warmup_batches = 10
    with torch.no_grad():
        with autocast_context:
            for i, batch in enumerate(loader):
                if i >= warmup_batches:
                    break
                inputs = {k: v.to(device) for k, v in batch.items()}
                _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Start GPU power sampling.
    sampler = GPUPowerSampler(gpu_index = gpu_index, sample_interval_s = power_sample_interval_s)
    sampler.start()

    all_predictions: List[int] = []
    all_labels: List[int] = []
    start_time = time.perf_counter()

    measured_batches = 0
    with torch.no_grad():
        with autocast_context:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if device.type == "cuda":
                    torch.cuda.synchronize()

                logits = model(**batch).logits

                if device.type == "cuda":
                    torch.cuda.synchronize()

                predictions = torch.argmax(logits, dim = -1).detach().cpu().numpy().tolist()
                labels = batch["labels"].detach().cpu().numpy().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels)

                measured_batches += 1
                if measured_batches >= num_inference_batches:
                    break

    end_time = time.perf_counter()
    sampler.stop()
    samples = sampler.get_samples()

    # Save power trace.
    save_power_trace_csv(samples, run_directory / "power_trace.csv")

    # Compute accuracy.
    accuracy = float(np.mean(np.array(all_predictions) == np.array(all_labels)))

    # Compute latency per example.
    total_examples = len(all_labels)
    total_time_s = end_time - start_time
    average_latency_per_example_ms = (total_time_s / total_examples) * 1000.0

    # Compute energy per example and per correct.
    total_energy_j = integrate_energy(samples)
    energy_per_example_j = total_energy_j / total_examples
    num_correct = int(np.sum(np.array(all_predictions) == np.array(all_labels)))
    energy_per_correct_j = total_energy_j / max(1, num_correct)

    peak_gpu_memory_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    with open(run_directory / "predictions.jsonl", "w", encoding="utf-8") as f:
        for prediction, label in zip(all_predictions, all_labels):
            f.write(json.dumps({"prediction": int(prediction), "label": int(label)}) + "\n")

    return InferenceBenchmarkResult(
        accuracy = accuracy,
        average_latency_per_example_ms = average_latency_per_example_ms,
        energy_per_example_j = energy_per_example_j,
        energy_per_correct_j = energy_per_correct_j,
        peak_gpu_memory_mb = peak_gpu_memory_mb,
    )   