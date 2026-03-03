from typing import Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

# Load SQuAD v1.1 and return tokenized train and validation splits with QA features
def load_and_tokenize_squad(
        model_name: str,
        max_sequence_length: int,
        doc_stride: int,
    ) -> Dict[str, object]:
    raw_datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    # Tokenize and create training features with start/end positions
    def prepare_train_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation = "only_second",
            max_length = max_sequence_length,
            stride = doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            padding = "max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answer, label CLS
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the token range corresponding to the context
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # If answer not fully inside the current span, label CLS
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            # Otherwise, move token_start_index and token_end_index to answer boundaries
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    # Tokenize and create validation features with example ids and offset mapping for postprocessing
    def prepare_validation_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation = "only_second",
            max_length = max_sequence_length,
            stride = doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            padding = "max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            # Set offsets to None for non-context tokens (question and special tokens)
            offsets = tokenized["offset_mapping"][i]
            tokenized["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offsets)
            ]

        return tokenized

    # Build tokenized splits
    train_features = raw_datasets["train"].map(
        prepare_train_features,
        batched = True,
        remove_columns = raw_datasets["train"].column_names,
    )

    validation_features = raw_datasets["validation"].map(
        prepare_validation_features,
        batched = True,
        remove_columns = raw_datasets["validation"].column_names,
    )

    train_features.set_format(type = "torch")
    validation_features.set_format(type = "torch")

    return {
        "tokenizer": tokenizer,
        "train_features": train_features,
        "validation_features": validation_features,
        "validation_examples": raw_datasets["validation"],
    }

# Load SQuAD v1.1 validation features for inference-only sweeps
def load_and_tokenize_squad_validation(
        model_name: str,
        max_sequence_length: int,
        doc_stride: int,
    ) -> Dict[str, object]:
    raw_datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

    # Tokenize validation features with overflow for postprocessing
    def prepare_validation_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation = "only_second",
            max_length = max_sequence_length,
            stride = doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            padding = "max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            offsets = tokenized["offset_mapping"][i]
            tokenized["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offsets)
            ]

        return tokenized

    validation_features = raw_datasets["validation"].map(
        prepare_validation_features,
        batched = True,
        remove_columns = raw_datasets["validation"].column_names,
    )

    validation_features.set_format(type = "torch")

    return {
        "tokenizer": tokenizer,
        "validation_features": validation_features,
        "validation_examples": raw_datasets["validation"],
    }