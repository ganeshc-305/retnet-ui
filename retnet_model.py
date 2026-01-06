"""
RetNet (Retentive Network) Model Implementation
A modern language model architecture with retention mechanisms.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetNetConfig:
    """Configuration for RetNet model architecture."""
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 6

    # Head dimensions
    head_dim_qk: int = 128
    head_dim_v: int = 256

    # FFN configuration
    ffn_dim: int = 1536

    # Sequence configuration
    max_seq_len: int = 512
    chunk_size: int = 128

    # Regularization
    dropout: float = 0.1

    # Training
    use_bias: bool = False

    # Numerical stability
    layernorm_eps: float = 1e-6

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim_qk, \
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) × head_dim_qk ({self.head_dim_qk})"
        assert self.max_seq_len % self.chunk_size == 0, \
            f"max_seq_len ({self.max_seq_len}) must be divisible by chunk_size ({self.chunk_size})"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LayerNorm(nn.Module):
    """Standard Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class XPos(nn.Module):
    """Extrapolatable Position Encoding."""
    
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0, downscale: bool = False) -> torch.Tensor:
        seq_len = x.shape[2]
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=x.dtype)
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        if downscale:
            rotated_x1 = x1 * cos + x2 * sin
            rotated_x2 = -x1 * sin + x2 * cos
        else:
            rotated_x1 = x1 * cos - x2 * sin
            rotated_x2 = x1 * sin + x2 * cos

        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.reshape(x.shape)


class Retention(nn.Module):
    """Single-head retention mechanism."""
    
    def __init__(self, d_model: int, head_dim_qk: int, head_dim_v: int, gamma: float, use_bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.gamma = gamma
        self.scale = head_dim_qk ** -0.5

        self.W_Q = nn.Linear(d_model, head_dim_qk, bias=use_bias)
        self.W_K = nn.Linear(d_model, head_dim_qk, bias=use_bias)
        self.W_V = nn.Linear(d_model, head_dim_v, bias=use_bias)
        self.xpos = XPos(head_dim_qk)

    def _build_decay_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay = self.gamma ** distance.clamp(min=0)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        return decay * mask

    def _normalize_decay_mask(self, D: torch.Tensor) -> torch.Tensor:
        row_sum = D.sum(dim=-1, keepdim=True)
        return D / torch.sqrt(row_sum.clamp(min=1.0))

    def parallel_forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self.xpos(Q.unsqueeze(1), downscale=False).squeeze(1)
        K = self.xpos(K.unsqueeze(1), downscale=True).squeeze(1)

        D = self._build_decay_mask(seq_len, device, dtype)
        if normalize:
            D = self._normalize_decay_mask(D)

        retention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        retention_scores = retention_scores * D.unsqueeze(0)

        if normalize:
            abs_sum = retention_scores.abs().sum(dim=-1, keepdim=True)
            retention_scores = retention_scores / torch.clamp(abs_sum, min=1.0)

        retention_scores = torch.clamp(retention_scores, min=-100.0, max=100.0)
        return torch.matmul(retention_scores, V)

    def recurrent_forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, position: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        if state is None:
            state = torch.zeros(batch_size, self.head_dim_qk, self.head_dim_v, device=device, dtype=dtype)

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self.xpos(Q.unsqueeze(1), offset=position, downscale=False).squeeze(1)
        K = self.xpos(K.unsqueeze(1), offset=position, downscale=True).squeeze(1)

        Q = Q.squeeze(1).squeeze(1)
        K = K.squeeze(1).squeeze(1)
        V = V.squeeze(1)

        kv = torch.einsum('bi,bj->bij', K, V)
        new_state = self.gamma * state + kv
        output = torch.einsum('bi,bij->bj', Q * self.scale, new_state)

        return output.unsqueeze(1), new_state

    def forward(self, x: torch.Tensor, mode: str = 'parallel', state: Optional[torch.Tensor] = None,
                position: int = 0, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mode == 'parallel':
            return self.parallel_forward(x), None
        elif mode == 'recurrent':
            return self.recurrent_forward(x, state, position)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class MultiScaleRetention(nn.Module):
    """Multi-head retention with multiple decay scales."""
    
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim_qk = config.head_dim_qk
        self.head_dim_v = config.head_dim_v
        self.d_model = config.d_model

        gammas = 1 - 2 ** (-5 - torch.arange(self.n_heads, dtype=torch.float32))
        self.register_buffer('gammas', gammas)

        self.retention_heads = nn.ModuleList([
            Retention(config.d_model, config.head_dim_qk, config.head_dim_v,
                      gammas[i].item(), config.use_bias)
            for i in range(self.n_heads)
        ])

        self.group_norm = nn.GroupNorm(self.n_heads, self.n_heads * config.head_dim_v, eps=config.layernorm_eps)
        self.W_G = nn.Linear(config.d_model, config.n_heads * config.head_dim_v, bias=config.use_bias)
        self.W_O = nn.Linear(config.n_heads * config.head_dim_v, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mode: str = 'parallel', states: Optional[List[torch.Tensor]] = None,
                position: int = 0, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        if states is None and mode != 'parallel':
            states = [None] * self.n_heads

        head_outputs = []
        new_states = [] if mode != 'parallel' else None

        for i, retention_head in enumerate(self.retention_heads):
            state = states[i] if states else None
            head_out, head_state = retention_head(x, mode=mode, state=state, position=position, chunk_size=chunk_size)
            head_outputs.append(head_out)
            if new_states is not None:
                new_states.append(head_state)

        Y = torch.cat(head_outputs, dim=-1)
        Y = Y.transpose(1, 2)
        Y = self.group_norm(Y)
        Y = Y.transpose(1, 2)

        gate = F.silu(self.W_G(x))  # Swish = SiLU
        Y = gate * Y

        output = self.W_O(Y)
        output = self.dropout(output)
        return output, new_states


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.W_1 = nn.Linear(config.d_model, config.ffn_dim, bias=config.use_bias)
        self.W_2 = nn.Linear(config.ffn_dim, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W_1(x)
        x = F.gelu(x)
        x = self.W_2(x)
        x = self.dropout(x)
        return x


class RetNetBlock(nn.Module):
    """Single RetNet transformer block."""
    
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.msr = MultiScaleRetention(config)
        self.ln_2 = LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, mode: str = 'parallel', states: Optional[List[torch.Tensor]] = None,
                position: int = 0, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        residual = x
        x = self.ln_1(x)
        msr_out, new_states = self.msr(x, mode=mode, states=states, position=position, chunk_size=chunk_size)
        x = residual + msr_out

        residual = x
        x = self.ln_2(x)
        x = residual + self.ffn(x)

        return x, new_states


class RetNet(nn.Module):
    """RetNet Language Model."""
    
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, mode: str = 'parallel',
                states: Optional[List[List[torch.Tensor]]] = None, position: int = 0,
                chunk_size: Optional[int] = None, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[List[torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        if states is None and mode != 'parallel':
            states = [[None] * self.config.n_heads for _ in range(self.config.n_layers)]

        new_states = [] if mode != 'parallel' else None

        for i, block in enumerate(self.blocks):
            block_states = states[i] if states else None
            x, block_new_states = block(x, mode=mode, states=block_states, position=position, chunk_size=chunk_size)
            if new_states is not None:
                new_states.append(block_new_states)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None, 
                 repetition_penalty: float = 1.0, use_recurrent: bool = False) -> torch.Tensor:
        """Generate text tokens autoregressively."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            x = generated if generated.size(1) <= self.config.max_seq_len else generated[:, -self.config.max_seq_len:]
            logits, _, _ = self.forward(x, mode='parallel')
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
