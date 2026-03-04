import argparse
import json
from typing import Dict, List
from transformers import AutoModelForSequenceClassification
from .configs import create_run_directory
from .data import load_and_tokenize_sst2_validation
from .evaluate_inference import benchmark_inference_model_with_precision
from .pareto import save_pareto_table, plot_energy_accuracy_combination, plot_energy_latency_combination

# Parse CLI arguments for selecting baseline model and combination sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--precisions", type = str, nargs = "+", default = ["fp32", "fp16"])
    parser.add_argument("--sequence_lengths", type = int, nargs = "+", default = [128, 96, 64, 48, 40, 32, 24, 16])
    parser.add_argument("--evaluation_batch_size", type = int, default = 64)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Run Phase 2: Fixed model, precision and sequence-length combination sweep
def run_phase_2_combination_sweep() -> None:
    args = parse_arguments()

    # Create run directory for outputs
    run_name = "phase_2_combination_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    # Persist sweep configuration for reproducibility
    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "precisions": args.precisions,
        "sequence_lengths": args.sequence_lengths,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "gpu_index": args.gpu_index,
        "dataset": "glue/sst2",
        "model_architecture": "bert-base-uncased finetuned classifier",
        "note": "Combination sweep nests precision and sequence length and logs one combined pareto.csv.",
    }
    with open(run_directory / "config.json", "w", encoding = "utf-8") as f:
        json.dump(sweep_configuration, f, indent = 4, sort_keys = True)

    pareto_rows: List[Dict[str, object]] = []

    # Cache tokenized validation sets by sequence length
    tokenized_validation_by_length: Dict[int, object] = {}

    # Sweep precision and sequence length
    for precision in args.precisions:
        precision = precision.lower().strip()

        for max_sequence_length in args.sequence_lengths:
            # Create per-measurement output directory
            sub_run_directory = run_directory / f"precision_{precision}" / f"max_sequence_length_{max_sequence_length}"
            sub_run_directory.mkdir(parents = True, exist_ok = False)

            # Tokenize validation once per sequence length
            if max_sequence_length not in tokenized_validation_by_length:
                tokenized_validation_by_length[max_sequence_length] = load_and_tokenize_sst2_validation(
                    model_name = "bert-base-uncased",
                    max_sequence_length = max_sequence_length,
                )
            tokenized_validation = tokenized_validation_by_length[max_sequence_length]

            # Load baseline model fresh for each measurement
            model = AutoModelForSequenceClassification.from_pretrained(args.baseline_model_directory)

            # Benchmark under specified precision
            inference_output = benchmark_inference_model_with_precision(
                run_directory = sub_run_directory,
                model = model,
                precision = precision,
                evaluation_dataset = tokenized_validation,
                evaluation_batch_size = args.evaluation_batch_size,
                num_inference_batches = args.num_inference_batches,
                power_sample_interval_s = args.power_sample_interval_s,
                gpu_index = args.gpu_index,
            )

            # Record one row per measurement
            pareto_rows.append({
                "label": f"{precision}-{max_sequence_length}",
                "precision": precision,
                "max_sequence_length": int(max_sequence_length),
                "accuracy": inference_output.accuracy,
                "energy_per_example_j": inference_output.energy_per_example_j,
                "energy_per_correct_j": inference_output.energy_per_correct_j,
                "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
                "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
            })

    # Save combined Pareto table and plots 
    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_accuracy_combination(pareto_csv_path, run_directory)
    _ = plot_energy_latency_combination(pareto_csv_path, run_directory)

    print(f"Phase 2 (combination) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_combination_sweep()