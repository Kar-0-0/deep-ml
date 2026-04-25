import torch
import torch.nn.functional as F
from typing import Tuple
import math 

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    q, k, v = X @ W_q, X @ W_k, X @ W_v

    return q, k, v

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1)) # (seq_len, seq__len)
    scores = F.softmax(scores, dim=-1)
    out = scores @ V

    return out



def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """

    head_size = K.size(1) // n_heads
    out = torch.cat([self_attention(Q.split(head_size, dim=-1)[i], K.split(head_size, -1)[i], V.split(head_size, -1)[i]) for i in range(n_heads)], dim=-1)

    return out
