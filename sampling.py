import torch


def top_k_sampling(logits, top_l, temp):
    logits = logits / temp
    topk_logits, topk_indices = torch.topk(logits)

    probs = F.softmax(topk_logits)
    sampled_index = torch.multinomial(probs, num_samples=1)
    token_id = topk_indices[sampled_index]
    return token_id


def top_p_sampling(logits, top_p, temp):
    logits = logits / temp
    probs = F.softmax(logits)

    sorted_logits, sorted_index = torch.sort(probs, descending=True)
    cummulative_probs = torch.cumsum(sorted_logits)
    cutoff_index = torch.searchsorted(cummulative_probs, p) + 1

    nucleus_logits, nucleus_index = (
        sorted_logits[:cutoff_index],
        sorted_logits[:cutoff_index],
    )
    nucleus_logits = nucleus_logits / nucleus_logits.sum()

    sampled_index = torch.multinomial(nucleus_logits, num_samples=1)
    token_id = nucleus_index[sampled_index]
    return token_id
