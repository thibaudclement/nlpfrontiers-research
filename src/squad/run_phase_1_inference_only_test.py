import argparse
import json
from pathlib import Path
from .configs import create_run_directory
from .data import load_and_tokenize_squad_validation
from .evaluate_inference import benchmark_inference_qa
from .pareto import save_pareto_table

# Parse CLI arguments for running inference-only baseline evaluation
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type=str, required=True)
    parser.add_argument("--max_sequence_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--evaluation_batch_size", type=int, default=16)

    # Use a large default so DataLoader naturally exhausts full validation features
    parser.add_argument("--num_inference_batches", type=int, default=100000)

    parser.add_argument("--power_sample_interval_s", type=float, default=0.05)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    return parser.parse_args()

# Run inference-only baseline evaluation for a pre-trained QA checkpoint
def run_inference_only_baseline() -> None:
    args = parse_arguments()

    # Create run directory for outputs
    run_directory = create_run_directory(base_directory="runs", run_name="phase_1_baseline_inference_only")

    # Save configuration for reproducibility
    config = {
        "run_name": "phase_1_baseline_inference_only",
        "baseline_model_directory": args.baseline_model_directory,
        "dataset": "squad_v1.1",
        "max_sequence_length": args.max_sequence_length,
        "doc_stride": args.doc_stride,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "n_best_size": args.n_best_size,
        "max_answer_length": args.max_answer_length,
        "gpu_index": args.gpu_index,
    }
    with open(run_directory / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    # Load validation features + examples
    data = load_and_tokenize_squad_validation(
        model_name="bert-base-uncased",
        max_sequence_length=args.max_sequence_length,
        doc_stride=args.doc_stride,
    )

    # Run inference benchmark (EM/F1 + energy/latency)
    inference_output = benchmark_inference_qa(
        run_directory=run_directory,
        model_directory=args.baseline_model_directory,
        validation_features=data["validation_features"],
        validation_examples=data["validation_examples"],
        evaluation_batch_size=args.evaluation_batch_size,
        num_inference_batches=args.num_inference_batches,
        power_sample_interval_s=args.power_sample_interval_s,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        gpu_index=args.gpu_index,
    )

    # Save metrics
    metrics = {
        "best_model_directory": args.baseline_model_directory,
        "inference_benchmark_metrics": {
            "exact_match": inference_output.exact_match,
            "f1": inference_output.f1,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        },
    }
    with open(run_directory / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    # Save baseline pareto row (single point)
    pareto_rows = [{
        "label": "bert_base_full_finetune",
        "max_sequence_length": args.max_sequence_length,
        "num_encoder_layers": 12,
        "exact_match": inference_output.exact_match,
        "f1": inference_output.f1,
        "energy_per_example_j": inference_output.energy_per_example_j,
        "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
        "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
    }]
    save_pareto_table(pareto_rows, run_directory)

    print("Inference-only baseline complete. Results saved to", run_directory)

if __name__ == "__main__":
    run_inference_only_baseline()