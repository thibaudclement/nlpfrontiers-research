import argparse
from pathlib import Path

from .configs import create_run_directory
from .data import load_and_tokenize_squad_validation
from .evaluate_inference import benchmark_inference_qa


# Parse CLI arguments for smoke-testing inference only
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_directory", type = str, required = True)
    parser.add_argument("--max_sequence_length", type = int, default = 384)
    parser.add_argument("--doc_stride", type = int, default = 128)
    parser.add_argument("--evaluation_batch_size", type = int, default = 16)
    parser.add_argument("--num_inference_batches", type = int, default = 5)
    parser.add_argument("--power_sample_interval_s", type = float, default = 0.05)
    parser.add_argument("--n_best_size", type = int, default = 20)
    parser.add_argument("--max_answer_length", type = int, default = 30)
    parser.add_argument("--gpu_index", type = int, default = 0)
    return parser.parse_args()

# Run a small inference-only benchmark to validate pipeline correctness
def run_smoke_test_inference() -> None:
    args = parse_arguments()

    # Create a run directory for smoke test artifacts
    run_directory = create_run_directory(base_directory = "runs", run_name = "squad_smoke_test_inference")

    # Load and tokenize SQuAD validation features for QA inference
    data = load_and_tokenize_squad_validation(
        model_name = "bert-base-uncased",
        max_sequence_length = args.max_sequence_length,
        doc_stride = args.doc_stride,
    )

    # Run inference benchmark on a small number of batches
    result = benchmark_inference_qa(
        run_directory = run_directory,
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

    print("Smoke test result:", result)
    print(f"Smoke test complete. Results saved to {run_directory}")

if __name__ == "__main__":
    run_smoke_test_inference()