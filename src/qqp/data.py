from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer

# Load QQP dataset from GLUE and return tokenized splits
def load_and_tokenize_qqp(model_name: str, max_sequence_length: int) -> Dict[str, object]:
    raw_datasets = load_dataset("glue", "qqp")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["question1"],
            batch["question2"],
            truncation = True,
            max_length = max_sequence_length,
            padding = "max_length",
        )

    tokenized_datasets = raw_datasets.map(tokenize_batch, batche = True)

    # Remove raw text and idx columns
    columns_to_remove = []
    for col in ["question1", "question2", "idx"]:
        if col in tokenized_datasets["train"].column_names:
            columns_to_remove.append(col)
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

    # Rename labels for Hugging Face Trainer
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type = "torch")

    return {
        "tokenizer": tokenizer,
        "train": tokenized_datasets["train"],
        "validation": tokenized_datasets["validation"],
    }

# Load QQP validation split and tokenize based on max sequence length
def load_and_tokenize_qqp_validation(model_name: str, max_sequence_length: int) -> object:
    raw_datasets = load_dataset("glue", "qqp")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["question1"],
            batch["question2"],
            truncation = True,
            max_length = max_sequence_length,
            padding = "max_length",
        )

    tokenized_validation = raw_datasets["validation"].map(tokenize_batch, batched = True)

    columns_to_remove = []
    for col in ["question1", "question2", "idx"]:
        if col in tokenized_validation.column_names:
            columns_to_remove.append(col)
    tokenized_validation = tokenized_validation.remove_columns(columns_to_remove)

    tokenized_validation = tokenized_validation.rename_column("label", "labels")
    tokenized_validation.set_format(type="torch")
    return tokenized_validation