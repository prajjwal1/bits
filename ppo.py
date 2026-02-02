# Batch size, L: sequence length, G: num of generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: [B*G, L]

# Get value predictions
value = value_net(completions)
