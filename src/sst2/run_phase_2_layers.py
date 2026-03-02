import argparse
import json
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification
from .configs import create_run_directory
from .data import load_and_tokenize_sst2_validation
from .evaluate_inference import benchmark_inference_model
from .pareto import save_pareto_table, plot_energy_accuracy_layers, plot_energy_latency_layers

# Parse CLI arguments for selecting baseline model and layer sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--num_layers", type = int, nargs = "+", default = [12, 10, 8, 6, 4, 2])
    parser.add_argument("--max_sequence_length", type = int, default = 128)
    parser.add_argument("--evaluation_batch_size", type = int, default = 64)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Load fine-tuned BERT classifier with only the first N layers
def load_bert_with_reduced_layers(model_directory: str, num_encoder_layers: int) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    
    # Validate layer count
    original_num_layers = len(model.bert.encoder.layer)
    if num_encoder_layers < 1 or num_encoder_layers > original_num_layers:
        raise ValueError(f"num_encoder_layers must be between 1 and {original_num_layers}, got {num_encoder_layers}")
    
    # Replace encoder layers with truncated version
    model.bert.encoder.layer = torch.nn.ModuleList(list(model.bert.encoder.layer[:num_encoder_layers]))
    return model

# Run Phase 2: Fixed model, varying number of layers at inference time
def run_phase_2_layers_sweep() -> None:
    args = parse_arguments()

    # Create run directory for outputs
    run_name = "phase_2_layers_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    # Persist sweep configuration for reproducibility
    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "num_layers": args.num_layers,
        "max_sequence_length": args.max_sequence_length,
        "evaluation_batch_size": args.evaluation_batch_size,
        "num_inference_batches": args.num_inference_batches,
        "power_sample_interval_s": args.power_sample_interval_s,
        "gpu_index": args.gpu_index,
        "dataset": "glue/sst2",
        "model_architecture": "bert-base-uncased finetuned classifier"
    }
    with open(run_directory / "config.json", "w", encoding = "utf-8") as f:
        json.dump(sweep_configuration, f, indent = 4, sort_keys = True)

    # Tokenize validation once at fixed max sequence length
    tokenized_validation = load_and_tokenize_sst2_validation(
        model_name = "bert-base-uncased",
        max_sequence_length = args.max_sequence_length,
    )

    pareto_rows: List[Dict[str, object]] = []

    # Sweep number of layers with fixed model weights
    for num_encoder_layers in args.num_layers:
        sub_run_directory = run_directory / f"num_layers_{num_encoder_layers}"
        sub_run_directory.mkdir(parents = True, exist_ok = False)

        reduced_model = load_bert_with_reduced_layers(
            model_directory = args.baseline_model_directory,
            num_encoder_layers = num_encoder_layers
        )

        inference_output = benchmark_inference_model(
            run_directory = sub_run_directory,
            model = reduced_model,
            evaluation_dataset = tokenized_validation,
            evaluation_batch_size = args.evaluation_batch_size,
            num_inference_batches = args.num_inference_batches,
            power_sample_interval_s = args.power_sample_interval_s,
            gpu_index = args.gpu_index,
        )

        pareto_rows.append({
            "label": str(num_encoder_layers),
            "num_encoder_layers": num_encoder_layers,
            "max_sequence_length": args.max_sequence_length,
            "accuracy": inference_output.accuracy,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        })

    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_accuracy_layers(pareto_csv_path, run_directory)
    _ = plot_energy_latency_layers(pareto_csv_path, run_directory)

    print(f"Phase 2 (layers) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_layers_sweep()