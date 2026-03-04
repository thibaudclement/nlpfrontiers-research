from __future__ import annotations
from typing import Any, Dict
import evaluate

# Compute SQuAD v2 metrics given predictions and raw evaluation dataset
def compute_squad_v2_metrics(
    predictions_by_example_id: Dict[str, str],
    raw_evaluation_dataset: Any,
) -> Dict[str, Any]:
    squad_v2_metric = evaluate.load("squad_v2")

    # Build predictions list expected by evaluate metric
    predictions = [
        {
            "id": example["id"],
            "prediction_text": predictions_by_example_id.get(example["id"], ""),
        }
        for example in raw_evaluation_dataset
    ]

    # Build references list expected by evaluate metric.
    references = [
        {
            "id": example["id"],
            "answers": example["answers"],
        }
        for example in raw_evaluation_dataset
    ]

    return squad_v2_metric.compute(predictions=predictions, references=references)