import argparse
import json
from typing import Dict, List
from .configs import create_run_directory
from .data import load_and_tokenize_squad_v1_validation
from .evaluate_inference import benchmark_inference_qa
from .pareto import (
    save_pareto_table,
    plot_energy_em_max_sequence_length,
    plot_energy_f1_max_sequence_length,
    plot_energy_latency_max_sequence_length,
)

# Parse CLI arguments for selecting baseline model and max sequence length sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--sequence_lengths", type = int, nargs = "+", default = [384, 320, 256, 224, 192, 160, 128])
    parser.add_argument("--doc_stride", type = int, default = 128)
    parser.add_argument("--evaluation_batch_size", type = int, default = 16)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--n_best_size", type = int, default = 20)
    parser.add_argument("--max_answer_length", type = int, default = 30)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Run Phase 2: Fixed model, varying max sequence length at inference time
def run_phase_2_sequence_length_sweep() -> None:
    args = parse_arguments()

    run_name = "phase_2_sequence_length_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "sequence_lengths": args.sequence_lengths,
        "doc_stride": args.doc_stride,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "n_best_size": args.n_best_size,
        "max_answer_length": args.max_answer_length,
        "gpu_index": args.gpu_index,
        "dataset": "squad",
        "model_architecture": "bert-base-uncased finetuned QA",
    }
    with open(run_directory / "config.json", "w", encoding = "utf-8") as f:
        json.dump(sweep_configuration, f, indent = 4, sort_keys = True)

    pareto_rows: List[Dict[str, object]] = []

    for max_sequence_length in args.sequence_lengths:
        data = load_and_tokenize_squad_v1_validation(
            model_name = "bert-base-uncased",
            max_sequence_length = max_sequence_length,
            doc_stride = args.doc_stride,
        )

        sub_run_directory = run_directory / f"max_sequence_length_{max_sequence_length}"
        sub_run_directory.mkdir(parents = True, exist_ok = False)

        inference_output = benchmark_inference_qa(
            run_directory = sub_run_directory,
            model_directory = args.baseline_model_directory,
            validation_features = data["validation_features"],
            validation_examples = data["validation_examples"],
            evaluation_batch_size = args.evaluation_batch_size,
            num_inference_batches = args.num_inference_batches,
            power_sample_interval_s = args.power_sample_interval_s,
            n_best_size = args.n_best_size,
            max_answer_length = args.max_answer_length,
            gpu_index = args.gpu_index,
        )

        pareto_rows.append({
            "label": f"max_sequence_length_{max_sequence_length}",
            "max_sequence_length": max_sequence_length,
            "exact_match": inference_output.exact_match,
            "f1": inference_output.f1,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        })

    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_em_max_sequence_length(pareto_csv_path, run_directory)
    _ = plot_energy_f1_max_sequence_length(pareto_csv_path, run_directory)
    _ = plot_energy_latency_max_sequence_length(pareto_csv_path, run_directory)

    print(f"Phase 2 (max sequence length) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_sequence_length_sweep()