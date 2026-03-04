import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Store experiment parameters for reproducibility
@dataclass(frozen = True)
class ExperimentConfig:
    run_name: str
    model_name: str = "bert-base-uncased"
    dataset_name: str = "squad"
    dataset_config: str = "v1"
    max_sequence_length: int = 384
    doc_stride: int = 128
    train_batch_size: int = 16
    evaluation_batch_size: int = 16
    learning_rate: float = 3e-5
    num_train_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    use_fp16: bool = False
    num_inference_batches: int = 100000
    power_sample_interval_s: float = 0.05
    n_best_size: int = 20
    max_answer_length: int = 30

# Create run directory for outputs
def create_run_directory(base_directory: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{run_name}"
    run_directory = Path(base_directory) / run_id
    run_directory.mkdir(parents = True, exist_ok = False)
    return run_directory

# Save configuration to disk for reproducibility
def save_config(config: ExperimentConfig, run_directory: Path) -> None:
    config_path = run_directory / "config.json"
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(asdict(config), f, indent = 4, sort_keys = True)