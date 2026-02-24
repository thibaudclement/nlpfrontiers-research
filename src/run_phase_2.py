import argparse
import json
from pathlib import Path
from typing import Dict, List
from .configs import create_run_directory, save_config, ExperimentConfig
from .data import load_and_tokenize_sst2_validation
from .evaluate_inference import benchmark_inference
from .pareto import save_pareto_table, plot_energy_accuracy_pareto_frontier

# Parse CLI arguments for selecting baseline model and sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--sequence_lengths", type = int, nargs = "+", default = [128, 96, 64, 32])
    parser.add_argument("--evaluation_batch_size", type = int, default = 64)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Run Phase 2: Fixed model, varying max sequence length at inference time
def run_phase_2_sequence_length_sweep() -> None:
    args = parse_arguments()

    # Create run directory for outputs
    run_name = "phase_2_sequence_length_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    # Persist sweep configuration for reproducibility
    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "sequence_lengths": args.sequence_lengths,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "gpu_index": args.gpu_index,
        "dataset": "glue/sst2",
        "model_architecture": "bert-base-uncased finetuned classifier"
    }
    with open(run_directory / "config.json", "w", encoding = "utf-8") as f:
        json.dump(sweep_configuration, f, indent = 4, sort_keys = True)

    pareto_rows: List[Dict[str, object]] = []

    # Sweep max sequence length with fixed model weights
    for max_sequence_length in args.sequence_lengths:
        # Tokenize validation data for given max sequence length
        tokenized_validation = load_and_tokenize_sst2_validation(
            model_name = "bert-base-uncased",
            max_sequence_length = max_sequence_length,
        )

        # Create subdirectory for this max sequence length configuration
        sub_run_directory = run_directory / f"max_sequence_length_{max_sequence_length}"
        sub_run_directory.mkdir(parents = True, exist_ok = False)

        # Benchmark inference for given max sequence length
        inference_output = benchmark_inference(
            run_directory = sub_run_directory,
            model_directory = args.baseline_model_directory,
            evaluation_dataset = tokenized_validation,
            evaluation_batch_size = args.evaluation_batch_size,
            num_inference_batches = args.num_inference_batches,
            power_sample_interval_s = args.power_sample_interval_s,
            gpu_index = args.gpu_index,
        )

        # Record one row per configuration for Pareto analysis
        pareto_rows.append({
            "label": f"max_sequence_length_{max_sequence_length}",
            "max_sequence_length": max_sequence_length,
            "accuracy": inference_output.accuracy,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        })

    # Save Pareto table and plot frontier
    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_accuracy_pareto_frontier(pareto_csv_path, run_directory)

    print(f"Phase 2 (max sequence length) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_sequence_length_sweep()

