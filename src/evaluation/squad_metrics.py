from __future__ import annotations
from typing import Any, Dict
import evaluate

# Compute SQuAD v2 metrics given predictions, no-answer probabilities, and raw evaluation dataset
def compute_squad_v2_metrics(
    predictions_by_example_id: Dict[str, str],
    no_answer_probability_by_example_id: Dict[str, float],
    raw_evaluation_dataset: Any,
) -> Dict[str, Any]:
    squad_v2_metric = evaluate.load("squad_v2")

    # Build predictions list expected by evaluate metric
    predictions = [
        {
            "id": str(example["id"]),
            "prediction_text": predictions_by_example_id.get(str(example["id"]), ""),
            "no_answer_probability": float(no_answer_probability_by_example_id.get(str(example["id"]), 0.0)),
        }
        for example in raw_evaluation_dataset
    ]

    # Build references list expected by evaluate metric
    references = [
        {
            "id": str(example["id"]),
            "answers": example["answers"],
        }
        for example in raw_evaluation_dataset
    ]

    return squad_v2_metric.compute(predictions=predictions, references=references)