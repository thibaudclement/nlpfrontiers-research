import argparse
import json
from typing import Dict, List
from transformers import AutoModelForSequenceClassification
from .configs import create_run_directory
from .data import load_and_tokenize_sst2_validation
from .evaluate_inference import benchmark_inference_model_with_precision
from .pareto import save_pareto_table, plot_energy_accuracy_precision, plot_energy_latency_precision

# Parse CLI arguments for selecting baseline model and precision sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--precisions", type = str, nargs = "+", default = ["fp32", "fp16", "fp8"])
    parser.add_argument("--max_sequence_length", type = int, default = 128)
    parser.add_argument("--evaluation_batch_size", type = int, default = 64)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--gpu_index", type = int, default = 0)
    parser.add_argument("--skip_failed_precisions", action = "store_true")
    return parser.parse_args()

# Run Phase 2: Fixed model, varying precision at inference time
def run_phase_2_precision_sweep() -> None:
    args = parse_arguments()

    run_name = "phase_2_precision_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "precisions": args.precisions,
        "max_sequence_length": args.max_sequence_length,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "gpu_index": args.gpu_index,
        "dataset": "glue/sst2",
        "model_architecture": "bert-base-uncased finetuned classifier",
        "note": "FP8 is best-effort and may be skipped if unsupported.",
    }
    with open(run_directory / "config.json", "w", encoding = "utf-8") as f:
        json.dump(sweep_configuration, f, indent = 4, sort_keys = True)

    tokenized_validation = load_and_tokenize_sst2_validation(
        model_name = "bert-base-uncased",
        max_sequence_length = args.max_sequence_length,
    )

    pareto_rows: List[Dict[str, object]] = []
    errors: Dict[str, str] = {}

    for precision in args.precisions:
        precision = precision.lower().strip()
        sub_run_directory = run_directory / f"precision_{precision}"
        sub_run_directory.mkdir(parents=True, exist_ok=False)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.baseline_model_directory)

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

            pareto_rows.append({
                "label": precision,
                "precision": precision,
                "max_sequence_length": args.max_sequence_length,
                "accuracy": inference_output.accuracy,
                "energy_per_example_j": inference_output.energy_per_example_j,
                "energy_per_correct_j": inference_output.energy_per_correct_j,
                "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
                "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
            })

        except Exception as e:
            errors[precision] = repr(e)
            if not args.skip_failed_precisions:
                raise

    if errors:
        with open(run_directory / "errors.json", "w", encoding = "utf-8") as f:
            json.dump(errors, f, indent = 4, sort_keys = True)

    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_accuracy_precision(pareto_csv_path, run_directory)
    _ = plot_energy_latency_precision(pareto_csv_path, run_directory)

    print(f"Phase 2 (precision) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_precision_sweep()