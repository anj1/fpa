import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import List, Optional

def create_inputs(
    b: int = 2,
    h: int = 4,
    t: int = 64,
    d_in: int = 32,
    d_v: int = 32,
    n_branches: int = 3,
    head_first: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    causal: bool = True,
    gating: bool = False,
    norm: bool = True,
    requires_grad: bool = False,
    seed: int = 42,
    std: float = 1.0
):
    """
    Creates a dictionary of inputs for the factorized_polynomial_attention function.

    Args:
        b (int): Batch size.
        h (int): Number of heads.
        t (int): Sequence length.
        d_in (int): Input head dimension for Q and K.
        d_v (int): Value head dimension for V.
        n_branches (int): The degree of the polynomial, which corresponds to the number
            of projection branches (n).
        branch_dims (Optional[List[int]]): A list of output dimensions for each
            projection branch. If None, a default is used. Its length must equal `n_branches`.
        head_first (bool): If True, uses the (batch, head, seq, dim) layout.
            This implementation only supports True.
        dtype (torch.dtype): The data type for the tensors.
        device (str): The device to place tensors on (e.g., 'cuda', 'cpu').
        causal (bool): Whether to apply a causal mask.
        normalize (bool): Whether to normalize the attention output.
        requires_grad (bool): If True, sets requires_grad=True for the generated tensors.
        seed (int): Random seed for reproducibility.
        std (float): Standard deviation for the random initialization.

    Returns:
        dict: A dictionary containing all the inputs for the FPA function.
    """
    torch.manual_seed(seed)

    shape_qkv = (b, h, t) if head_first else (b, t, h)

    Q = torch.randn(size=(*shape_qkv, d_in), dtype=dtype, device=device) * std
    K = torch.randn(size=(*shape_qkv, d_in), dtype=dtype, device=device) * std
    V = torch.randn(size=(*shape_qkv, d_v), dtype=dtype, device=device) * std

    # Shape is (b, h, n, d) for head_first, and (b, n, h, d) otherwise
    shape_g_diag = (b, h, n_branches, d_in) if head_first else (b, n_branches, h, d_in)
    
    G_diagonal = torch.randn(
       *shape_g_diag, dtype=dtype, device=device
    ) * std

    if requires_grad:
        Q, K, V, G_diagonal = tree_map(lambda x: x.requires_grad_(True), (Q, K, V, G_diagonal))

    log_G = F.logsigmoid(torch.rand(size=shape_qkv, dtype=torch.float32, device=device)) if gating else None

    return dict(
        Q=Q,
        K=K,
        V=V,
        log_G =log_G,
        G_diagonal=G_diagonal,
        causal=causal,
        norm=norm,
        head_first=head_first
    )

def input_properties(
    b: int = 2,
    h: int = 4,
    t: int = 64,
    d_in: int = 32,
    d_v: int = 48,
    n_branches: int = 3,
    head_first: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    gating = False,
    **kwargs
):
    """
    Returns a dictionary describing the properties (shape, dtype, device) of the input tensors.
    """
    shape_qkv = (b, h, t) if head_first else (b, t, h)
    shape_g_diag = (b, h, n_branches, d_in) if head_first else (b, n_branches, h, d_in)
    g_diag_props = (shape_g_diag, dtype, device)

    shape_qkv = (b, h, t)
    g_diag_props = ((b, h, n_branches, d_in), dtype, device)

    return dict(
        Q=((*shape_qkv, d_in), dtype, device),
        K=((*shape_qkv, d_in), dtype, device),
        V=((*shape_qkv, d_v), dtype, device),
        G_diagonal=g_diag_props
    ) | ({'log_G': (shape_qkv, torch.float32, device)} if gating else {})


def output_properties(
    b: int = 2,
    h: int = 4,
    t: int = 64,
    d_v: int = 48,
    head_first: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    **kwargs
):
    """
    Returns a dictionary describing the properties (shape, dtype, device) of the output tensor.
    """
    shape_out = (b, h, t) if head_first else (b, t, h)
    return dict(
        Y=((*shape_out, d_v), dtype, device),
    )
