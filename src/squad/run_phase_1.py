import json
from .configs import ExperimentConfig, create_run_directory, save_config
from .data import load_and_tokenize_squad_v1
from .evaluate_inference import benchmark_inference_qa
from .finetune import finetune_baseline
from .pareto import save_pareto_table

# Run Phase 1: Baseline finetuning and QA inference benchmarking
def run_phase_1() -> None:
    config = ExperimentConfig(run_name = "phase_1_baseline")
    run_directory = create_run_directory(base_directory = "runs", run_name = config.run_name)
    save_config(config, run_directory)

    data = load_and_tokenize_squad_v1(
        model_name = config.model_name,
        max_sequence_length = config.max_sequence_length,
        doc_stride = config.doc_stride,
    )

    # Fine-tune baseline QA model
    finetune_output = finetune_baseline(
        run_directory = run_directory,
        model_name = config.model_name,
        train_dataset = data["train_features"],
        evaluation_dataset = data["validation_features"],
        train_batch_size = config.train_batch_size,
        evaluation_batch_size = config.evaluation_batch_size,
        learning_rate = config.learning_rate,
        num_train_epochs = config.num_train_epochs,
        weight_decay = config.weight_decay,
        warmup_ratio = config.warmup_ratio,
        seed = config.seed,
        use_fp16 = config.use_fp16,
    )

    # Benchmark inference on best model (EM/F1)
    inference_output = benchmark_inference_qa(
        run_directory = run_directory,
        model_directory = finetune_output["best_model_directory"],
        validation_features = data["validation_features"],
        validation_examples = data["validation_examples"],
        evaluation_batch_size = config.evaluation_batch_size,
        num_inference_batches = config.num_inference_batches,
        power_sample_interval_s = config.power_sample_interval_s,
        n_best_size = config.n_best_size,
        max_answer_length = config.max_answer_length,
        gpu_index = 0,
    )

    # Save metrics for analysis
    metrics = {
        "trainer_evaluation_metrics": finetune_output["evaluation_metrics"],
        "inference_benchmark_metrics": {
            "exact_match": inference_output.exact_match,
            "f1": inference_output.f1,
            "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
            "energy_per_example_j": inference_output.energy_per_example_j,
            "energy_per_correct_j": inference_output.energy_per_correct_j,
            "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
        },
        "best_model_directory": finetune_output["best_model_directory"],
    }
    with open(run_directory / "metrics.json", "w", encoding = "utf-8") as f:
        json.dump(metrics, f, indent = 4, sort_keys = True)

    # Create initial Pareto table with baseline results
    pareto_rows = [{
        "label": "bert_base_full_finetune",
        "max_sequence_length": config.max_sequence_length,
        "num_encoder_layers": 12,
        "exact_match": inference_output.exact_match,
        "f1": inference_output.f1,
        "energy_per_example_j": inference_output.energy_per_example_j,
        "average_latency_per_example_ms": inference_output.average_latency_per_example_ms,
        "peak_gpu_memory_mb": inference_output.peak_gpu_memory_mb,
    }]
    save_pareto_table(pareto_rows, run_directory)

    print(f"Phase 1 complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_phase_1()