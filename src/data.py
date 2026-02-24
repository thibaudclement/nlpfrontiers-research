from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer

# Load SST-2 dataset from GLUE and return tokenized splits
def load_and_tokenize_sst2(model_name: str, max_sequence_length: int) -> Dict[str, object]:
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation = True,
            max_length = max_sequence_length,
            padding = "max_length",
        )
    
    tokenized_datasets = raw_datasets.map(tokenize_batch, batched = True)

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type = "torch")

    return {
        "tokenizer": tokenizer,
        "train": tokenized_datasets["train"],
        "validation": tokenized_datasets["validation"],
    }

# Load SST-2 validation split and tokenize based on max sequence length
def load_and_tokenized_sst2_validation(model_name: str, max_sequence_length: int) -> object:
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation = True,
            max_length = max_sequence_length,
            padding = "max_length",
        )
    
    tokenized_validation = raw_datasets["validation"].map(tokenize_batch, batched = True)
    tokenized_validation = tokenized_validation.remove_columns(["sentence", "idx"])
    tokenized_validation = tokenized_validation.rename_column("label", "labels")
    tokenized_validation.set_format(type = "torch")
    return tokenized_validation