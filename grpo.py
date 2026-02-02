import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class GRPOConfig:
    """
        group_size: number of outputs to sample from each prompt
        clip_epsilon: PPO-style clipping parameter
        kl_coef: coefficient for KL divergence penalty
        temperature: sampling temperature for generation
        max_length: maximum seq length for generation
    """
    group_size: int = 4
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    temperature: float = 1.0
    max_length: int = 512
    num_epochs: int = 1
    learning_rate: float = 1e-6


class GRPOTrainer:
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        reward_fn: Callable,
        config: GRPOConfig = None
    ):
        self.policy = policy_model
        self.ref_policy = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or GRPOConfig()

        # freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

    def sample_group(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = prompt_ids.shape[0]
        G = self.config.group_size

        all_outputs = []
        all_masks = []

        with torch.no_grad():
            for _ in range(G):
                outputs = self.policy.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_length,
                    do_sample=True,
                    temperatrue=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                all_outputs.append(outputs)

                mask = (outputs != self.tokenizer.pad_token_id).float()
                all_masks.append(mask)

        output_ids = torch.stack(all_outputs, dim=1).view(batch_size * G, -1)
        output_masks = torch.stack(all_masks, dim=1).view(batch_size * G, -1)
        return output_ids, output_masks

    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits
        # shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:, :]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
