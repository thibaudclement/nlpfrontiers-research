from pathlib import Path
from typing import Dict
from transformers import (
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    set_seed
)

# Fine-tune a baseline QA model and return evaluation metrics and saved model path
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

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    data_collator = DefaultDataCollator()

    num_update_steps_per_epoch = (len(train_dataset) + train_batch_size - 1) // train_batch_size
    total_training_steps = int(num_update_steps_per_epoch * num_train_epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)

    # Configure Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir = str(run_directory / "checkpoints"),
        eval_strategy = "no",
        save_strategy = "epoch",
        learning_rate = learning_rate,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = evaluation_batch_size,
        num_train_epochs = num_train_epochs,
        weight_decay = weight_decay,
        warmup_steps = warmup_steps,
        logging_steps = 50,
        load_best_model_at_end = False,
        fp16 = use_fp16,
        report_to = "none",
        seed = seed,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = evaluation_dataset,
        data_collator = data_collator,
    )

    trainer.train()

    # Save the trained model for inference benchmarking
    save_model_directory = run_directory / "best_model"
    trainer.save_model(str(save_model_directory))

    return {
        "evaluation_metrics": {},
        "best_model_directory": str(save_model_directory)
    }