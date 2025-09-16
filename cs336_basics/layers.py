import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None, optional): device to store the parameters on
            dtype (torch.dtype | None, optional): data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Initialize weights: N(0, 2/(d_in+d_out)) truncated to [-3σ, 3σ]
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transform: returns x @ W^T.
        """
        return x.matmul(self.weight.transpose(-1, -2))


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Embedding class that maps integer token IDs to a vector space of d_model, i.e., 
        performs a embedding lookup using table of size (vocab_size, d_model).

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialization: N(0, 1) truncated to [-3, 3].
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs by indexing.

        Args:
            token_ids (torch.Tensor): (batch_size, sequence_length) with the token IDs to lookup
        
        Returns:
            torch.Tensor: (batch_size, sequence_length, d_model) with the embedding vectors for the given token IDs
        """
        # Notes:
        # When you index a 2D tensor T[rows, cols] with an integer index tensor I for rows, 
        # the result appends the remaining column dimension. i.e., (B, S) indices -> (B, S, d_model).
        #
        # Indexing weight with a LongTensor of shape (batch_size, sequence_length) 
        # performs a batched row lookup along the first dimension (vocab_size).
        # i.e., 
        # weight[token_ids.long()] maps each integer ID in token_ids to its corresponding row in weight, 
        # preserving the first two dims (batch_size, sequence_length) and adding the embedding dim d_model.
        return self.weight[token_ids.long()]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Root Mean Square Layer Normalization (RMSNorm).

        Args:
            d_model (int): Hidden dimension of the model (size of the last dimension).
            eps (float): Epsilon value for numerical stability.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Tensor of the same shape as input.
        """
        # Upcast input to torch.float32 to prevent overflow when square the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute root mean square over the last dimension
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        y = (x / rms) * self.weight.to(torch.float32)
        return y.to(in_dtype)


class Softmax(nn.Module):
    def __init__(self, i: int, device=None):
        """
        Apply softmax to the i-th dimension of the input tensor, 
        returning a tensor of the same shape,
        but its i-th dimension will have a normalized probability distribution.

        Args:
            dim_i (int): Dimension to apply softmax to.
            device (torch.device | None, optional): device hint (unused).
        """
        super().__init__()
        self.i = i
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically-stable softmax along dimension `self.dim_i`.
        Uses in-place operations to reduce memory traffic.
        """
        # Subtract max for numerical stability, then exponentiate in-place
        y = x - torch.amax(x, dim=self.i, keepdim=True)
        y = y.exp_()
        # Normalize in-place
        y /= torch.sum(y, dim=self.i, keepdim=True)
        return y


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        """
        SwiGLU positionwise feed-forward network, composed of a SiLU activation
        function and a GLU.

        Args:
            d_model (int): Input/output hidden size.
            d_ff (int | None): Inner feed-forward size. If None, set to approximately
                8/3 * d_model and round to a multiple of 64.
            device (torch.device | None): Device for parameters.
            dtype (torch.dtype | None): Dtype for parameters.
        """
        super().__init__()
        if d_ff is None:
            approx = 8.0 * d_model / 3.0
            d_ff = max(64, int(round(approx / 64.0) * 64))
        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear projections for SwiGLU
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU.

        Computes: (SiLU(x @ W1^T) ⊙ (x @ W3^T)) @ W2^T

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        a = self.w1(x)
        b = self.w3(x)
        # SiLU using sigmoid for numerical stability
        a = a * torch.sigmoid(a)
        y = a * b
        return self.w2(y)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): maximum sequence length that will be inputted
            device (torch.device | None, optional): device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        if self.d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE (pairs of dimensions are rotated).")

        # Precompute angular frequencies and corresponding cos/sin for all positions.
        # Angular frequencies θ^(−2k/d_k)
        exponents = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32) / float(self.d_k)
        ang_freq = torch.pow(torch.tensor(self.theta, device=device, dtype=torch.float32), -exponents)
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * ang_freq[None, :]  # (max_seq_len, d_k//2)

        # Register as non-persistent buffers to avoid bloating the state dict
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoRE to the input tensor and return a tensor of the same shape. 
        Use token positions to slice (possibly precomputed) cos and sin tensors along the sequence dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) 
                specifying the token positions of x along the sequence dimension.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_k)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Lookup cos/sin for the given positions. Shape: (..., seq_len, d_k//2)
        cos = self.cos_cached[token_positions].to(torch.float32)
        sin = self.sin_cached[token_positions].to(torch.float32)

        # Split last dim into even/odd parts to apply 2D rotations per pair.
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Re-interleave to original layout
        y = torch.empty_like(x)
        y[..., ::2] = x_even * cos - x_odd * sin
        y[..., 1::2] = x_even * sin + x_odd * cos

        return y.to(in_dtype)
