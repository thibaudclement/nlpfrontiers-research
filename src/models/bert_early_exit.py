from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import BertConfig, BertForQuestionAnswering, BertModel, BertPreTrainedModel
from transformers.utils import ModelOutput

# Store outputs for full-pass multi-exit question answering
@dataclass
class EarlyExitQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    exit_start_logits: Optional[Tuple[torch.Tensor, ...]] = None
    exit_end_logits: Optional[Tuple[torch.Tensor, ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None

# Store outputs for true dynamic single-example early-exit inference
@dataclass
class DynamicEarlyExitQuestionAnsweringOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    selected_exit_layer: int
    selected_exit_index: int
    executed_layer_count: int
    confidence: float

# Compute normalized entropy-based confidence from start/end logits
def compute_entropy_based_exit_confidence(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
) -> torch.Tensor:
    start_probabilities = torch.softmax(start_logits, dim=-1)
    end_probabilities = torch.softmax(end_logits, dim=-1)

    start_log_probabilities = torch.log(start_probabilities.clamp_min(1e-12))
    end_log_probabilities = torch.log(end_probabilities.clamp_min(1e-12))

    start_entropy = -(start_probabilities * start_log_probabilities).sum(dim=-1)
    end_entropy = -(end_probabilities * end_log_probabilities).sum(dim=-1)

    sequence_length = start_logits.size(-1)
    normalization_denominator = torch.log(
        torch.tensor(
            max(2, int(sequence_length)),
            dtype=start_logits.dtype,
            device=start_logits.device,
        )
    )

    normalized_start_entropy = start_entropy / normalization_denominator
    normalized_end_entropy = end_entropy / normalization_denominator

    confidence = 1.0 - 0.5 * (normalized_start_entropy + normalized_end_entropy)
    return confidence

# Implement BERT QA with intermediate exits for robust training and evaluation
class BertForQuestionAnsweringEarlyExit(BertPreTrainedModel):
    # Initialize backbone and one QA head per configured exit layer
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.early_exit_layers = list(getattr(config, "early_exit_layers", [4, 8, 12]))
        self.early_exit_loss_weights = list(
            getattr(config, "early_exit_loss_weights", [1.0] * len(self.early_exit_layers))
        )

        if len(self.early_exit_layers) == 0:
            raise ValueError("early_exit_layers must contain at least one layer index")

        if len(self.early_exit_layers) != len(self.early_exit_loss_weights):
            raise ValueError("early_exit_layers and early_exit_loss_weights must have the same length")

        if sorted(self.early_exit_layers) != list(self.early_exit_layers):
            raise ValueError("early_exit_layers must be sorted in increasing order")

        if len(set(self.early_exit_layers)) != len(self.early_exit_layers):
            raise ValueError("early_exit_layers must not contain duplicates")

        for early_exit_layer in self.early_exit_layers:
            if int(early_exit_layer) < 1 or int(early_exit_layer) > int(config.num_hidden_layers):
                raise ValueError(
                    f"Each early exit layer must be in [1, {config.num_hidden_layers}], "
                    f"but got {early_exit_layer}"
                )

        self.exit_qa_outputs = nn.ModuleList(
            [nn.Linear(config.hidden_size, 2) for _ in self.early_exit_layers]
        )

        self.post_init()

    # Compute standard extractive QA loss for one exit
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

        loss_function = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_function(start_logits, start_positions)
        end_loss = loss_function(end_logits, end_positions)

        return 0.5 * (start_loss + end_loss)

    # Project one hidden-state tensor into start and end logits
    def compute_exit_logits(
        self,
        hidden_states: torch.Tensor,
        exit_head_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.exit_qa_outputs[int(exit_head_index)](hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    # Run a robust full forward pass and read intermediate hidden states for exits
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
        # Force hidden states on because exit heads depend on them
        backbone_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states_by_layer = backbone_outputs.hidden_states
        exit_start_logits = []
        exit_end_logits = []
        per_exit_losses = []

        # Read encoder-layer hidden states and apply each exit head
        for exit_head_index, exit_layer in enumerate(self.early_exit_layers):
            exit_hidden_states = hidden_states_by_layer[int(exit_layer)]
            start_logits, end_logits = self.compute_exit_logits(
                hidden_states=exit_hidden_states,
                exit_head_index=exit_head_index,
            )

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

        final_start_logits = exit_start_logits[-1]
        final_end_logits = exit_end_logits[-1]

        total_loss = None
        if len(per_exit_losses) > 0:
            weighted_losses = [
                float(loss_weight) * per_exit_loss
                for loss_weight, per_exit_loss in zip(self.early_exit_loss_weights, per_exit_losses)
            ]
            total_loss = sum(weighted_losses) / sum(float(weight) for weight in self.early_exit_loss_weights)

        if not return_dict:
            output = (
                final_start_logits,
                final_end_logits,
                tuple(exit_start_logits),
                tuple(exit_end_logits),
                backbone_outputs.hidden_states if bool(output_hidden_states) else None,
                backbone_outputs.attentions,
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return EarlyExitQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=final_start_logits,
            end_logits=final_end_logits,
            exit_start_logits=tuple(exit_start_logits),
            exit_end_logits=tuple(exit_end_logits),
            hidden_states=backbone_outputs.hidden_states if bool(output_hidden_states) else None,
            attentions=backbone_outputs.attentions,
        )

    # Run true dynamic early-exit inference for a single example or single feature window
    def run_dynamic_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        early_exit_confidence_threshold: float = 1.01,
    ) -> DynamicEarlyExitQuestionAnsweringOutput:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                "run_dynamic_early_exit expects a batch of size 1 so that executed depth matches "
                "selected depth exactly"
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Print device information once to verify that dynamic inference runs on CUDA.
        if not hasattr(self, "_has_printed_dynamic_device_info"):
            print(
                "[dynamic] tensor devices: "
                f"input_ids={input_ids.device}, "
                f"attention_mask={attention_mask.device}, "
                f"token_type_ids={token_type_ids.device}, "
                f"model_device={next(self.parameters()).device}",
                flush=True,
            )
            self._has_printed_dynamic_device_info = True

        # Build same extended mask structure that BERT encoder layers expect
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
        )

        # Compute embeddings once, then step through encoder layers manually
        hidden_states = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        exit_layer_to_head_index = {
            int(exit_layer): head_index
            for head_index, exit_layer in enumerate(self.early_exit_layers)
        }

        selected_start_logits = None
        selected_end_logits = None
        selected_exit_layer = int(self.early_exit_layers[-1])
        selected_exit_index = len(self.early_exit_layers) - 1
        selected_confidence = 0.0

        # Step through encoder and stop as soon as threshold is met
        for layer_index, layer_module in enumerate(self.bert.encoder.layer, start=1):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=None,
                output_attentions=False,
            )

            # Support both tensor-returning and tuple-returning layer implementations
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if layer_index not in exit_layer_to_head_index:
                continue

            exit_head_index = exit_layer_to_head_index[layer_index]
            start_logits, end_logits = self.compute_exit_logits(
                hidden_states=hidden_states,
                exit_head_index=exit_head_index,
            )

            confidence_tensor = compute_entropy_based_exit_confidence(
                start_logits=start_logits,
                end_logits=end_logits,
            )
            confidence_value = float(confidence_tensor.item())

            selected_start_logits = start_logits
            selected_end_logits = end_logits
            selected_exit_layer = int(layer_index)
            selected_exit_index = int(exit_head_index)
            selected_confidence = confidence_value

            if confidence_value >= float(early_exit_confidence_threshold):
                return DynamicEarlyExitQuestionAnsweringOutput(
                    start_logits=selected_start_logits,
                    end_logits=selected_end_logits,
                    selected_exit_layer=selected_exit_layer,
                    selected_exit_index=selected_exit_index,
                    executed_layer_count=selected_exit_layer,
                    confidence=selected_confidence,
                )

        return DynamicEarlyExitQuestionAnsweringOutput(
            start_logits=selected_start_logits,
            end_logits=selected_end_logits,
            selected_exit_layer=selected_exit_layer,
            selected_exit_index=selected_exit_index,
            executed_layer_count=selected_exit_layer,
            confidence=selected_confidence,
        )

# Initialize early-exit model from standard fine-tuned BERT QA checkpoint
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

    # Copy encoder and embedding weights from fine-tuned base QA model
    early_exit_model.bert.load_state_dict(base_model.bert.state_dict())

    # Initialize every exit head from final QA head of base model
    base_qa_head_state = base_model.qa_outputs.state_dict()
    for exit_head in early_exit_model.exit_qa_outputs:
        exit_head.load_state_dict(base_qa_head_state)

    return early_exit_model