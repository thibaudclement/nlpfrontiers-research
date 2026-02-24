import json
import time
from dataclasses
from pathlib
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from .measure import GPUPowerSampler, integrate_energy, save_power_trace_csv

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
    num_inferance_batches: int,
    power_sample_interval_s: float,
) -> InferenceBenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    model.to(device)
    model.eval()

    loader = DataLoader(evaluation_dataset, batch_size = evaluation_batch_size, shuffle = false)

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
            _ = model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Start GPU power sampling
    sampler = GPUPowerSampler(gpu_index = 0, sample_interval_s = power_sample_interval_s)
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
            if measured_batches >= num_inferance_batches:
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

    # Save preductions for analysis
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