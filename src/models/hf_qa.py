from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

@dataclass
class QuestionAnsweringArtifacts:
    tokenizer: any
    model: any

# Load Hugging Face extractive QA model and tokenizer
def load_question_answering_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
) -> QuestionAnsweringArtifacts:
    tokenizer_identifier = tokenizer_name_or_path or model_name_or_path

    # Load tokenizer with fast implementation where available
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier, use_fast=True)

    # Load question answering head on top of base model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

    return QuestionAnsweringArtifacts(tokenizer=tokenizer, model=model)