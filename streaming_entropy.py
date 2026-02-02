# computing entropy, in a streaming fashion
# only one pass

# p_i = e^{li} / Z where Z = \sum_j e^{li}
# H = -pi \sum_j log pi
# log p_i = li - log(Z)
# substitute p_i and log p_i in H
# H = \sum_i (e^{li}/Z) * (li - log(Z))
# H = logZ - 1/Z \sum e^{li} li


def streaming_entropy(blocks):
    blocks = np.asarray(blocks)
    running_max = 0.0
    sum_exp, sum_exp_logit = 0.0, 0.0

    for block in blocks:
        block_max = np.max(block)
        new_max = np.max(block_max, running_max)

        if block_max != np.inf:
            scale = np.exp(running_max - new_max)
            sum_exp *= scale
            sum_exp_logit *= scale

        exp = np.exp(block - new_max)
        sum_exp += np.sum(exp)
        sum_exp_logit += np.sum(exp * block)

        running_max = new_max

    logZ = np.log(sum_exp) + running_max
    S = logZ - sum_exp_logit / sum_exp
    return S
