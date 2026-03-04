from __future__ import annotations
from typing import Any

# Count non-padding tokens in tokenized feature dataset using attention_mask
def count_non_padding_tokens_in_feature_dataset(feature_dataset: Any) -> int:
    total_tokens = 0

    # Iterate over rows (attention_mask is typically a list of ints per row)
    for row in feature_dataset:
        attention_mask = row.get("attention_mask")
        if attention_mask is not None:
            total_tokens += int(sum(attention_mask))

    return int(total_tokens)