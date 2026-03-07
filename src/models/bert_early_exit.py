from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import BertConfig, BertForQuestionAnswering, BertModel, BertPreTrainedModel
from transformers.utils import ModelOutput

# Output container for multi-exit BERT QA
@dataclass
class EarlyExitQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    exit_start_logits: Optional[Tuple[torch.Tensor, ...]] = None
    exit_end_logits: Optional[Tuple[torch.Tensor, ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None

# BERT QA model with intermediate early-exit heads
class BertForQuestionAnsweringEarlyExit(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        # Read early-exit metadata from config
        self.early_exit_layers = list(getattr(config, "early_exit_layers", [4, 8, 12]))
        self.early_exit_loss_weights = list(
            getattr(config, "early_exit_loss_weights", [1.0] * len(self.early_exit_layers))
        )

        if len(self.early_exit_layers) != len(self.early_exit_loss_weights):
            raise ValueError("early_exit_layers and early_exit_loss_weights must have the same length")

        # One QA head per exit layer
        self.exit_qa_outputs = nn.ModuleList(
            [nn.Linear(config.hidden_size, 2) for _ in self.early_exit_layers]
        )

        self.post_init()

    # Compute QA loss for one exit head
    def compute_question_answering_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> torch.Tensor:
        ignored_index = start_logits.size(1)

        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        return 0.5 * (start_loss + end_loss)

    # Forward pass with logits at all configured exits
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> EarlyExitQuestionAnsweringModelOutput:
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = bert_outputs.hidden_states

        exit_start_logits = []
        exit_end_logits = []
        per_exit_losses = []

        # Hidden states (index 0 = embeddings, index i = output of encoder layer i)
        for exit_head_index, exit_layer in enumerate(self.early_exit_layers):
            sequence_output = hidden_states[int(exit_layer)]
            logits = self.exit_qa_outputs[exit_head_index](sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            exit_start_logits.append(start_logits)
            exit_end_logits.append(end_logits)

            if start_positions is not None and end_positions is not None:
                per_exit_losses.append(
                    self.compute_question_answering_loss(
                        start_logits=start_logits,
                        end_logits=end_logits,
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )
                )

        # Final exit is the latest configured layer
        final_start_logits = exit_start_logits[-1]
        final_end_logits = exit_end_logits[-1]

        total_loss = None
        if len(per_exit_losses) > 0:
            weighted_losses = []
            for loss_weight, per_exit_loss in zip(self.early_exit_loss_weights, per_exit_losses):
                weighted_losses.append(float(loss_weight) * per_exit_loss)

            total_loss = sum(weighted_losses) / sum(float(weight) for weight in self.early_exit_loss_weights)

        if not return_dict:
            output = (
                final_start_logits,
                final_end_logits,
                tuple(exit_start_logits),
                tuple(exit_end_logits),
                bert_outputs.hidden_states,
                bert_outputs.attentions,
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return EarlyExitQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=final_start_logits,
            end_logits=final_end_logits,
            exit_start_logits=tuple(exit_start_logits),
            exit_end_logits=tuple(exit_end_logits),
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )

# Initialize early-exit BERT QA model from standard fine-tuned BERT QA checkpoint
def initialize_early_exit_model_from_base_checkpoint(
    base_checkpoint_path: str,
    early_exit_layers: list[int],
    early_exit_loss_weights: Optional[list[float]] = None,
) -> BertForQuestionAnsweringEarlyExit:
    base_model = BertForQuestionAnswering.from_pretrained(base_checkpoint_path)
    config = BertConfig.from_pretrained(base_checkpoint_path)

    config.early_exit_layers = list(early_exit_layers)
    config.early_exit_loss_weights = (
        list(early_exit_loss_weights)
        if early_exit_loss_weights is not None
        else [1.0] * len(early_exit_layers)
    )

    early_exit_model = BertForQuestionAnsweringEarlyExit(config)

    # Copy encoder weights
    early_exit_model.bert.load_state_dict(base_model.bert.state_dict())

    # Initialize every exit head from final QA head
    base_qa_head_state = base_model.qa_outputs.state_dict()
    for exit_head in early_exit_model.exit_qa_outputs:
        exit_head.load_state_dict(base_qa_head_state)

    return early_exit_model