import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

seq_length = 2048
embedding_dim = 512
batch = 4
num_heads = 8

class Embedding(torch.nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.embedding_func = nn.Linear(1, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor):
        input_ids = input_ids[:, :, None]
        return self.embedding_func(input_ids)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding // self.num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask = None):
        # Step 1: Linear transformations for Q, K, V
        batch_size = query.size(0)
        seq_length = query.size(1)
        Q = self.query(query)
        K = self.query(key)
        V = self.query(value)

        # Step 2: Split into multiple heads
        Q = Q.view(batch, seq_length, self.num_heads, self.embedding // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(batch, seq_length, self.num_heads, self.embedding // self.num_heads).permute(0, 2, 1, 3)
        V = V.view(batch, seq_length, self.num_heads, self.embedding // self.num_heads).permute(0, 2, 1, 3)

        # Step 3: Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2))
        print(f"The attention_scores has shape: {attention_scores.shape}")
        attention_weights = attention_scores / self.head_dim ** 0.5
        output = F.softmax(attention_weights, dim=-1)
        x = torch.matmul(output, V)
        print(f"The x has shape: {x.shape}")

        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        print(f"The x has shape: {x.shape}")
        x = x.view(batch_size, seq_length, -1)

        return self.fc_out(x)

        

input_ids = torch.randn(batch, seq_length)
print(f"The input ids has shape: {input_ids.shape}")
Embedding_forward = Embedding(embedding_dim, seq_length)
embedding_output = Embedding_forward(input_ids)
print(f"The embedding outputs has shape: {embedding_output.shape}")
MHA_forward = MultiHeadAttention(embedding_dim, num_heads)
MHA_output = MHA_forward(embedding_output, embedding_output, embedding_output)
print(f"The MHA_output has shape: {MHA_output.shape}")