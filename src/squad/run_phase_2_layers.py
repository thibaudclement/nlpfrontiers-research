import argparse
import json
from typing import Dict, List
import torch
from transformers import AutoModelForQuestionAnswering
from .configs import create_run_directory
from .data import load_and_tokenize_squad_v1_validation
from .evaluate_inference import benchmark_inference_qa
from .pareto import (
    save_pareto_table,
    plot_energy_em_layers,
    plot_energy_f1_layers,
    plot_energy_latency_layers,
)

# Parse CLI arguments for selecting baseline model and layer sweep settings
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--num_layers", type = int, nargs = "+", default = [12, 10, 8, 6, 4, 2])
    parser.add_argument("--max_sequence_length", type = int, default = 384)
    parser.add_argument("--doc_stride", type = int, default = 128)
    parser.add_argument("--evaluation_batch_size", type = int, default = 16)
    parser.add_argument("--num_inference_batches", type = int, default = 200)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--n_best_size", type = int, default = 20)
    parser.add_argument("--max_answer_length", type = int, default = 30)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Load fine-tuned BERT QA model with only the first N layers
def load_bert_qa_with_reduced_layers(model_directory: str, num_encoder_layers: int) -> AutoModelForQuestionAnswering:
    model = AutoModelForQuestionAnswering.from_pretrained(model_directory)

    original_num_layers = len(model.bert.encoder.layer)
    if num_encoder_layers < 1 or num_encoder_layers > original_num_layers:
        raise ValueError(f"num_encoder_layers must be between 1 and {original_num_layers}, got {num_encoder_layers}")

    model.bert.encoder.layer = torch.nn.ModuleList(list(model.bert.encoder.layer[:num_encoder_layers]))
    return model

# Run Phase 2: Fixed model, varying number of layers at inference time
def run_phase_2_layers_sweep() -> None:
    args = parse_arguments()

    run_name = "phase_2_layers_sweep"
    run_directory = create_run_directory(base_directory = "runs", run_name = run_name)

    sweep_configuration = {
        "run_name": run_name,
        "baseline_model_directory": args.baseline_model_directory,
        "num_layers": args.num_layers,
        "max_sequence_length": args.max_sequence_length,
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

    data = load_and_tokenize_squad_v1_validation(
        model_name = "bert-base-uncased",
        max_sequence_length = args.max_sequence_length,
        doc_stride = args.doc_stride,
    )

    pareto_rows: List[Dict[str, object]] = []

    for num_encoder_layers in args.num_layers:
        sub_run_directory = run_directory / f"num_layers_{num_encoder_layers}"
        sub_run_directory.mkdir(parents = True, exist_ok = False)

        reduced_model = load_bert_qa_with_reduced_layers(
            model_directory = args.baseline_model_directory,
            num_encoder_layers = num_encoder_layers,
        )

        # Save reduced model temporarily to disk for consistent loading path
        reduced_dir = sub_run_directory / "reduced_model"
        reduced_model.save_pretrained(str(reduced_dir))

        inference_output = benchmark_inference_qa(
            run_directory = sub_run_directory,
            model_directory = str(reduced_dir),
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
            "label": str(num_encoder_layers),
            "num_encoder_layers": num_encoder_layers,
            "max_sequence_length": args.max_sequence_length,
            "exact_match": inference_output.exact_match,
            "f1": inference_output.f1,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        })

    pareto_csv_path = save_pareto_table(rows = pareto_rows, run_directory = run_directory)
    _ = plot_energy_em_layers(pareto_csv_path, run_directory)
    _ = plot_energy_f1_layers(pareto_csv_path, run_directory)
    _ = plot_energy_latency_layers(pareto_csv_path, run_directory)

    print(f"Phase 2 (layers) complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_2_layers_sweep()