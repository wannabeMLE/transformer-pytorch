import torch
import torch.nn.functional as F

# Simulate: batch of 4 source sentences, each with 6 tokens
batch_size = 4
seq_len = 6
d_model = 8

# Simulate token IDs (usually from tokenizer)
encoder_input = torch.tensor([
    [3, 7, 2, 5, 1, 0],   # Sentence 1
    [8, 6, 5, 0, 0, 0],   # Sentence 2
    [4, 3, 2, 1, 0, 0],   # Sentence 3
    [9, 5, 2, 6, 3, 1]    # Sentence 4
])

# Embedding layer: vocab_size=10, embedding_dim=d_model
embedding = torch.nn.Embedding(10, d_model)

# Embedded input → shape: [batch_size, seq_len, d_model]
embedded = embedding(encoder_input)

# Use same values for Q, K, V
Q = K = V = embedded  # [4, 6, 8]

# Compute attention scores: Q @ K^T → [B, S, S]
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)

# Softmax across last dimension
attention_weights = F.softmax(scores, dim=-1)

# Compute context vectors
context = torch.matmul(attention_weights, V)  # shape: [B, S, d_model]

# Print result shape
print("Context shape:", context.shape)
print("Attention weights (after softmax):\n", attention_weights)
print("Difference between context and original embedding:\n", context - embedded)
print("Context vectors (attention_weights @ V):\n", context)
