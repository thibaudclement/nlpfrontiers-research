from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

# Load raw SQuAD v2 splits (optionally truncated for harness validation)
def load_raw_squad_v2_splits(
    huggingface_dataset_id: str,
    train_split_name: str,
    evaluation_split_name: str,
    maximum_train_examples: Optional[int] = None,
    maximum_evaluation_examples: Optional[int] = None,
):
    dataset_dict = load_dataset(huggingface_dataset_id)
    raw_train_split = dataset_dict[train_split_name]
    raw_evaluation_split = dataset_dict[evaluation_split_name]

    # Optionally truncate datasets for quick harness validation runs
    if maximum_train_examples is not None:
        raw_train_split = raw_train_split.select(range(min(maximum_train_examples, len(raw_train_split))))
    if maximum_evaluation_examples is not None:
        raw_evaluation_split = raw_evaluation_split.select(range(min(maximum_evaluation_examples, len(raw_evaluation_split))))

    return raw_train_split, raw_evaluation_split

# Convert raw examples into tokenized training features for extractive QA
def prepare_squad_v2_training_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    document_stride: int,
    pad_to_maximum_length: bool,
):
    # Strip leading whitespace from questions
    questions = [question.lstrip() for question in examples["question"]]

    # Tokenize with sliding window over the context
    tokenized_features = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=maximum_sequence_length,
        stride=document_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_maximum_length else False,
    )

    # Map tokenized spans back to original examples
    overflow_to_sample_mapping = tokenized_features.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_features.pop("offset_mapping")
    start_positions: List[int] = []
    end_positions: List[int] = []

    # Label each tokenized feature with start/end token positions (or CLS for no-answer)
    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_features["input_ids"][feature_index]
        cls_token_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_features.sequence_ids(feature_index)
        example_index = overflow_to_sample_mapping[feature_index]
        answers = examples["answers"][example_index]

        # Handle unanswerable questions (SQuAD v2) by labeling CLS
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_token_index)
            end_positions.append(cls_token_index)
            continue

        answer_start_character = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_character = answer_start_character + len(answer_text)

        # Identify token span corresponding to context tokens
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer is not fully inside current context window, label CLS
        if not (
            offsets[token_start_index][0] <= answer_start_character
            and offsets[token_end_index][1] >= answer_end_character
        ):
            start_positions.append(cls_token_index)
            end_positions.append(cls_token_index)
            continue

        # Otherwise, move token_start_index forward to first token that starts after answer_start
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_character:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # Move token_end_index backward to last token that ends before answer_end
        while offsets[token_end_index][1] >= answer_end_character:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_features["start_positions"] = start_positions
    tokenized_features["end_positions"] = end_positions
    return tokenized_features


# Convert raw examples into tokenized evaluation features for extractive QA
def prepare_squad_v2_evaluation_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    document_stride: int,
    pad_to_maximum_length: bool,
):
    # Strip leading whitespace from questions
    questions = [question.lstrip() for question in examples["question"]]

    # Tokenize with sliding window over context
    tokenized_features = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=maximum_sequence_length,
        stride=document_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_maximum_length else False,
    )

    # Map tokenized spans back to original examples
    overflow_to_sample_mapping = tokenized_features.pop("overflow_to_sample_mapping")
    tokenized_features["example_id"] = []

    # Store example_id and set offsets outside context to None
    for feature_index in range(len(tokenized_features["input_ids"])):
        example_index = overflow_to_sample_mapping[feature_index]
        tokenized_features["example_id"].append(examples["id"][example_index])

        sequence_ids = tokenized_features.sequence_ids(feature_index)
        offsets = tokenized_features["offset_mapping"][feature_index]
        tokenized_features["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_position] == 1 else None
            for token_position, offset in enumerate(offsets)
        ]

    return tokenized_features

# Postprocess raw start/end logits into final text predictions and no-answer probabilities
def postprocess_squad_v2_predictions(
    raw_examples: Any,
    tokenized_features: Any,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer: PreTrainedTokenizerBase,
    n_best_size: int = 20,
    maximum_answer_length: int = 30,
    null_score_difference_threshold: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    all_start_logits, all_end_logits = raw_predictions

    # Map example id to index for grouping features by example
    example_id_to_index = {example_id: index for index, example_id in enumerate(raw_examples["id"])}

    # Group feature indices by example index
    feature_indices_per_example: Dict[int, List[int]] = {}
    for feature_index, feature in enumerate(tokenized_features):
        example_index = example_id_to_index[feature["example_id"]]
        feature_indices_per_example.setdefault(example_index, []).append(feature_index)

    predicted_text_by_example_id: Dict[str, str] = {}
    predicted_no_answer_probability_by_example_id: Dict[str, float] = {}

    # Select the best answer candidate across all context windows per example
    for example_index, example in enumerate(raw_examples):
        feature_indices = feature_indices_per_example.get(example_index, [])
        context_text = example["context"]

        minimum_null_score: Optional[float] = None
        valid_answer_candidates: List[Dict[str, Any]] = []

        # Collect candidate spans across all context windows
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = tokenized_features[feature_index]["offset_mapping"]

            # Use CLS token score as the null (no-answer) score baseline
            cls_token_position = 0
            null_score = float(start_logits[cls_token_position] + end_logits[cls_token_position])
            if minimum_null_score is None or null_score < minimum_null_score:
                minimum_null_score = null_score

            # Select top-n start and end indices
            best_start_indices = np.argsort(start_logits)[-n_best_size:][::-1]
            best_end_indices = np.argsort(end_logits)[-n_best_size:][::-1]

            # Enumerate plausible (start, end) spans
            for start_index in best_start_indices:
                for end_index in best_end_indices:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue

                    answer_length = end_index - start_index + 1
                    if answer_length > maximum_answer_length:
                        continue

                    start_char, _ = offset_mapping[start_index]
                    _, end_char = offset_mapping[end_index]
                    predicted_text = context_text[start_char:end_char]
                    candidate_score = float(start_logits[start_index] + end_logits[end_index])

                    valid_answer_candidates.append({"score": candidate_score, "text": predicted_text})

        # Choose best non-null candidate (if any)
        if valid_answer_candidates:
            best_candidate = max(valid_answer_candidates, key=lambda item: item["score"])
            best_non_null_text = str(best_candidate["text"])
            best_non_null_score = float(best_candidate["score"])
        else:
            best_non_null_text = ""
            best_non_null_score = -1e9

        # Compute score difference
        effective_null_score = float(minimum_null_score) if minimum_null_score is not None else -1e9
        score_difference = best_non_null_score - effective_null_score

        # Compute no-answer probability from score difference
        no_answer_probability = convert_score_difference_to_no_answer_probability(score_difference)

        # Decide whether to output empty answer based on threshold
        example_id = str(example["id"])
        predicted_no_answer_probability_by_example_id[example_id] = float(no_answer_probability)

        # Apply SQuAD v2 null decision threshold
        if score_difference < null_score_difference_threshold:
            predicted_text_by_example_id[example_id] = ""
        else:
            predicted_text_by_example_id[example_id] = best_non_null_text

    return predicted_text_by_example_id, predicted_no_answer_probability_by_example_id