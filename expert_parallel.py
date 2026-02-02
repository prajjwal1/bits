from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from bits.model import Expert
from torch import nn


@dataclass
class EPConfig:
    n_routed_experts: int = 64  # total number of experts
    n_activated_experts: int = 6  # topK experts per token
    ep_size: int = 4  # number of experts in EP group
    dim: int = 2048
    moe_inter_dim: int = 1408

    @property
    def n_local_experts(self) -> int:
        return self.n_routed_experts // self.ep_size


def create_ep_group(world_size: int, ep_size: int) -> dist.ProcessGroup:
    """
    Create Expert Parallel groups using dp2ep sharding

    With world_size=8 and ep_size=4:
    - EP Group 0: [0, 1, 2, 3]
    - EP Group 1: [4, 5, 6, 7]
    """
    rank = dist.get_rank()

    ep_groups = []
    for idx in range(world_size // ep_size):
        ranks = list(range(idx * ep_size, (idx + 1) * ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            curr_group = group
            curr_rank = ranks.index(rank)
        ep_groups.append(group)
    return curr_group, curr_rank


class MoEWithAll2All(nn.Module):
    """
    Implement MoE layer with EP using all2all

    all2all exchanges data so that each GPU processes
    only the tokens meant for its expert
    """

    def __init__(self, config: EPConfig, ep_group: dist.ProcessGroup, ep_rank: int):
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = config.ep_size

        self.n_local_experts = config.n_local_experts
        self.experts_start_idx = ep_rank * self.n_local_experts
        self.experts_end_idx = (ep_rank + 1) * self.n_local_experts

        # gate is replicated on all ranks
        self.gate = nn.Linear(config.dim, config.n_routed_experts, bias=False)

        # create local experts only
        self.experts = nn.ModuleList(
            [
                Expert(config.dim, config.moe_inter_dim)
                for _ in range(self.n_local_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, dim)  # [num_tokens, dim]

        # Step 1: Gate - determine which expert each token goes to
        router_logits = self.gate(x_flat)  # [num_tokens, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # TopK routing
        topk_weights, topk_indices = torch.topk(
            router_probs, k=self.config.n_activated_experts, dim=-1
        )  # [num_tokens, topk]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Step 2: Prepare tokens for all2all dispatch
        # need to send each token to the GPU that owns the expert
        dispatch_tokens, dispatch_metadata = self._prepare_dispatch(
            x_flat, topk_indices, topk_weights
        )

        # Step 3: all2all dispatch - send tokens to expert owning GPUs
        received_tokens, recv_counts = self._all2all_dispatch(
            dispatch_tokens, dispatch_metadata
        )

        # Step 4: local expert computation
        expert_outputs = self._compute_local_experts(received_tokens, dispatch_metadata)

        # Step 5: all2all combine - return results to original GPUs
        combined_output = self._all2all_combine(expert_outputs, dispatch_metadata)

        # Step 6: weighted sum of expert outputs
        output = self._aggregate_outputs(
            combined_output,
            topk_weights,
            topk_indices,
            num_tokens,
            dispatch_metadata,
        )

        return output.view(batch_size, seq_len, dim)

    def _prepare_dispatch(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        topk_indices: torch.Tensor,  # [num_tokens, topk]
        topk_weights: torch.Tensor,  # [num_tokens, topk]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Group tokens by destination GPU.
        Each token might go to multiple experts on different GPUs.
        """
        num_tokens, topk = topk_indices.shape

        # determine which GPU each expert lives on
        expert_to_gpu = (
            topk_indices // self.n_local_experts
        )  # map expert ID -> GPU ID; [num_tokens, topk]

        # count how many token expert pairs go to each GPU
        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=x.device)
        for gpu_id in range(self.ep_size):
            send_counts[gpu_id] = (expert_to_gpu == gpu_id).sum()

        # create dispatch order: group by destination GPU
        dispatch_order = []
        token_expert_pairs = (
            []
        )  # Track which (token, expert_idx) each dispatched token represents

        for gpu_id in range(self.ep_size):
            mask = expert_to_gpu == gpu_id
            token_ids, expert_slots = torch.where(mask)
            dispatch_order.append(token_ids)
            # store local expert index within the destination GPU
            local_expert_ids = (
                topk_indices[token_ids, expert_slots] % self.n_local_experts
            )
            token_expert_pairs.append((token_ids, expert_slots, local_expert_ids))

        dispatch_order = torch.cat(dispatch_order)
        dispatch_tokens = x[dispatch_order]  # [total_dispatched, dim]

        metadata = {
            "send_counts": send_counts,
            "dispatch_order": dispatch_order,
            "token_expert_pairs": token_expert_pairs,
            "num_tokens": num_tokens,
            "topk": topk,
        }
        return dispatch_tokens, metadata

    def _all2all_dispatch(
        self, dispatched_tokens, metadata: dict  # [total_dispatched, dim],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The core all2all communication:

        1. Exchange counts: Each GPU tells others how many tokens it's sending
        2. Allocate receive buffer: based on received counts
        3. all2all tokens: send token tensors grouped by destination
        4. all2all expert IDs: send which local expert each token needs
        """
        send_counts = metadata["send_counts"]
        dim = dispatched_tokens.shape[-1]

        # Exchange counts so each GPU knows how many tokens it will receive
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # Prepare receive buffer
        total_recv = recv_counts.sum().item()
        received_tokens = torch.empty(
            total_recv,
            dim,
            dtype=dispatched_tokens.dtype,
            device=dispatched_tokens.device,
        )

        # Split tensor according to counts
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # all2all operation
        input_list = list(dispatched_tokens.split(send_splits))
        output_list = list(received_tokens.split(recv_splits))
        dist.all_to_all(output_list, input_list, group=self.ep_group)

        # also need to send local_expert_ids so receiving GPU knows which expert to use
        # gather local expert IDs from metadata
        local_expert_ids = torch.cat(
            [pair[2] for pair in metadata["token_expert_pairs"]]
        )  # [total_dispatched]
        received_expert_ids = torch.empty(
            total_recv, dtype=local_expert_ids.dtype, device=local_expert_ids.device
        )
        id_input_list = list(local_expert_ids.split(send_splits))
        id_output_list = list(received_expert_ids.split(recv_splits))
        dist.all_to_all(id_output_list, id_input_list, group=self.ep_group)

        # Store recv_counts in metadata for the combine step
        metadata["recv_counts"] = recv_counts
        metadata["received_expert_ids"] = received_expert_ids
        return torch.cat(output_list), recv_counts

    def _compute_local_experts(
        self,
        received_tokens: torch.Tensor,  # [total_received, dim]
        metadata,
    ) -> torch.Tensor:
        """
        Process received tokens through local experts
        Each token has an associated local_expert_id telling us
        which of our local experts should process it
        """
        output = torch.zeros_like(received_tokens)

        # get expert IDs
        received_expert_ids = metadata["received_expert_ids"]

        for local_idx in range(self.n_local_experts):
            mask = received_expert_ids == local_idx
            if mask.any():
                expert = self.experts[local_idx]
                output[mask] = expert(received_tokens[mask])
        return output

    def _all2all_combine(
        self, expert_outputs: torch.Tensor, metadata: dict
    ) -> torch.Tensor:
        """Reverse all2all: return each expert outputs to original GPUs"""
        recv_counts = metadata["recv_counts"]
        send_counts = metadata["send_counts"]
        dim = expert_outputs.shape[-1]

        # Receive buffer for combine
        total_tokens_send_back = send_counts.sum().item()
        combined = torch.empty(
            total_tokens_send_back,
            dim,
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )

        # reverse splits (send becomes recv, recv becomes send)
        send_splits = recv_counts.tolist()
        recv_splits = send_counts.tolist()

        input_list = list(expert_outputs.split(send_splits))
        output_list = list(combined.split(recv_splits))
        dist.all_to_all(output_list, input_list, group=self.ep_group)

        return torch.cat(output_list)

    def _aggregate_outputs(
        self,
        combined_output: torch.Tensor,  # [total_dispatched, dim]
        topk_weights: torch.Tensor,  # [num_tokens, topk]
        topk_indices: torch.Tensor,  # unused, kept for API
        num_tokens: int,
        metadata,
    ) -> torch.Tensor:
        """
        Aggregate weighted expert outputs back to original token positions
        """
        dim = combined_output.shape[-1]
        output = torch.zeros(
            num_tokens, dim, device=combined_output.device, dtype=combined_output.dtype
        )

        # need to reverse the dispatch order permutation
        # and apply topk_weights to the corresponding outputs

        token_expert_pairs = metadata["token_expert_pairs"]

        idx = 0
        for gpu_id in range(self.ep_size):
            token_ids, expert_slots, _ = token_expert_pairs[gpu_id]
            n = len(token_ids)
            weights = topk_weights[token_ids, expert_slots]
            output[token_ids] += weights.unsqueeze(-1) * combined_output[idx : idx + n]
            idx += n
        return output


if __name__ == "__main__":
    import os

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    assert world_size > 1

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config = EPConfig(
        n_routed_experts=64,
        n_activated_experts=6,
        ep_size=world_size,
        dim=2048,
        moe_inter_dim=1408,
    )

    ep_group, ep_rank = create_ep_group(world_size, world_size)

    model = MoEWithAll2All(config, ep_group, ep_rank).to(device, dtype=torch.bfloat16)

    x = torch.randn(4, 512, config.dim, device=device, dtype=torch.bfloat16)

    out = model(x)

    if dist.is_initialized():
        dist.destroy_process_group()
