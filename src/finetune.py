from pathlib import Path
from typing import Dict
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# Fine-tune a baseline model and return evaluation metrics and saved model path
def finetune_baseline(
        run_directory: Path,
        model_name: str,
        train_dataset: object,
        evaluation_dataset: object,
        train_batch_size: int,
        evaluation_batch_size: int,
        learning_rate: float,
        num_train_epochs: int,
        weight_decay: float,
        warmup_ratio: float,
        seed: int,
        use_fp16: bool,
    ) -> Dict[str, object]:
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    accuracy_metric = evaluate.load("accuracy")

    # Compute accuracy from logits and labels
    def compute_metrics(evalulation_predictions):
        logits, labels = evalulation_predictions
        preds = np.argmax(logits, axis = -1)
        return accuracy_metric.compute(predictions = preds, references = labels)
    
    # Configure Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir = str(run_directory / "checkpoints"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = learning_rate,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = evaluation_batch_size,
        num_train_epochs = num_train_epochs,
        weight_decay = weight_decay,
        warmup_ratio = warmup_ratio,
        logging_steps = 50,
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        greater_is_better = True,
        fp16 = use_fp16,
        report_to = "none",
        seed = seed,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = evaluation_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    evaluation_metrics = trainer.evaluate()

    # Save the best model for inference benchmarking
    save_model_directory = run_directory / "best_model"
    trainer.save_model(str(save_model_directory))

    return {
        "evaluation_metrics": evaluation_metrics,
        "best_model_directory": str(save_model_directory)
    }