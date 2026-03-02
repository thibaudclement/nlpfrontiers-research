import json
from .configs import ExperimentConfig, create_run_directory, save_config
from .data import load_and_tokenize_sst2
from .evaluate_inference import benchmark_inference
from .finetune import finetune_baseline
from .pareto import save_pareto_table, plot_energy_accuracy_pareto_frontier

# Run Phase 1: Baseline Finetuning and Inference Benchmarking
def run_phase_1() -> None:
    config = ExperimentConfig(run_name = "phase_1_baseline")
    run_directory = create_run_directory(base_directory = "runs", run_name = config.run_name)
    save_config(config, run_directory)
    data = load_and_tokenize_sst2(model_name = config.model_name, max_sequence_length = config.max_sequence_length)

    # Finetune baseline model
    finetune_output = finetune_baseline(
        run_directory = run_directory,
        model_name = config.model_name,
        train_dataset = data["train"],
        evaluation_dataset = data["validation"],
        train_batch_size = config.train_batch_size,
        evaluation_batch_size = config.evaluation_batch_size,
        learning_rate = config.learning_rate,
        num_train_epochs = config.num_train_epochs,
        weight_decay = config.weight_decay,
        warmup_ratio = config.warmup_ratio,
        seed = config.seed,
        use_fp16 = config.use_fp16,
    )

    # Benchmark inference on best model
    inference_output = benchmark_inference(
        run_directory = run_directory,
        model_directory = finetune_output["best_model_directory"],
        evaluation_dataset = data["validation"],
        evaluation_batch_size = config.evaluation_batch_size,
        num_inference_batches = config.num_inference_batches,
        power_sample_interval_s = config.power_sample_interval_s,
        gpu_index = 0
    )

    # Save metrics for Pareto analysis
    metrics = {
        "trainer_evaluation_metrics": finetune_output["evaluation_metrics"],
        "inference_benchmark_metrics": {
            "accuracy": inference_output.accuracy,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb
        },
        "best_model_directory": finetune_output["best_model_directory"],
    }
    with open(run_directory / "metrics.json", "w", encoding = "utf-8") as f:
        json.dump(metrics, f, indent = 4, sort_keys = True)

    # Create initial Pareto table with baseline results
    pareto_rows = [{
        "label": "bert_base_full_finetune",
        "max_sequence_length": config.max_sequence_length,
        "layers": 12,
        "accuracy": inference_output.accuracy,
        "energy_per_example_j": inference_output.energy_per_example_j,
        "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
        "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb
    }]

    pareto_csv_path = save_pareto_table(pareto_rows, run_directory)
    _ = plot_energy_accuracy_pareto_frontier(pareto_csv_path, run_directory)

    print(f"Phase 1 complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_1()