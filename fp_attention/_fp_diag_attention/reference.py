import torch
import torch.nn.functional as F
import math

# Epsilon for numerical stability
ε = 1e-6

def fpa_diagonal_attention(Q, K, V, G_diagonals, log_G, n_branches, r=1, w=1, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):
    """
    Reference implementation for Factorized Polynomial Attention (FPA) with Diagonal Factors.

    Args:
        Q, K, V: Query, Key, and Value tensors.
        G_diagonals (torch.Tensor): The diagonal vectors for the G matrices of each branch.
                                    Shape should be (batch, heads, n_branches, dim).
        log_G (torch.Tensor): Gating biases.
        n_branches (int): The number of branches in FPA, equivalent to the polynomial degree.
        r, w (int): Grouping factors for queries and keys.
        causal (bool): Whether to apply a causal mask.
        head_first (bool): If True, input shape is (b, h, t, d). Otherwise, (b, t, h, d).
        scale (float): Scaling factor for the attention scores.
        norm (bool): If True, normalize the output by the sum of attention weights.
        use_log2 (bool): If True, use log2/exp2 instead of natural log/exp.
    """
    if head_first:
        b, hq, ctx, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
    else:
        b, ctx, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]

    assert hq % r == 0, "hq must be divisible by r"
    assert hk % w == 0, "hk must be divisible by w"
    assert hq // r == hk // w, "hq // r must be equal to hk // w"
    h = hq // r
    
    if head_first:
        assert G_diagonals.shape == (b, h, n_branches, d)
    else:
        # Transpose G if needed, assuming it's passed in (b, n_branches, h, d) or similar
        # For simplicity, we'll assume it's correctly shaped for non-head-first
        assert G_diagonals.shape == (b, n_branches, h, d)
        G_diagonals = G_diagonals.transpose(1, 2)  # -> (b, h, n_branches, d)


    if log_G is not None:
        if head_first:
            assert log_G.shape == (b, h, ctx)
        else:
            assert log_G.shape == (b, ctx, h)
            log_G = log_G.transpose(1, 2) # (b, h, ctx)
    
    if head_first:
        Q = Q.view(b, h, ctx * r, d)
        K = K.view(b, h, ctx * w, d)
        V = V.view(b, h, ctx * w, e)
    else:
        Q = Q.view(b, ctx * r, h, d).transpose(1, 2)
        K = K.view(b, ctx * w, h, d).transpose(1, 2)
        V = V.view(b, ctx * w, h, e).transpose(1, 2)
    
    exp = torch.exp if not use_log2 else torch.exp2
    log = torch.log if not use_log2 else torch.log2

    _qidx = torch.arange(ctx*r, device=Q.device).unsqueeze(1)
    _kidx = torch.arange(ctx*w, device=K.device).unsqueeze(0)
    
    s = torch.ones(b, h, ctx * r, ctx * w, device=Q.device, dtype=Q.dtype)

    for i in range(n_branches):
        # Shape: (b, h, 1, d) for broadcasting with Q
        g_i = G_diagonals[:, :, i, :].unsqueeze(2)
        Q_proj = Q * g_i
        s_branch = torch.matmul(Q_proj, K.transpose(2, 3))
        s = s * s_branch
        
    s = s * scale

    m = (_qidx // r) >= (_kidx // w) if causal else torch.ones_like(s, dtype=torch.bool)
    signs = torch.sign(s)
    
    s = torch.where(m, log(s.abs() + ε), -float("inf"))
    
    if log_G is not None:
        s = s + (log_G.repeat_interleave(r, dim=2)[..., :, None] - log_G.repeat_interleave(w, dim=2)[..., None, :])
    
    rowmax = torch.max(s, dim=-1, keepdim=True).values.detach()
    
    p = exp(s - rowmax).to(V.dtype) * signs
    
    l = torch.sum(p, dim=-1).to(torch.float32) + ε
    o = torch.matmul(p, V)
    
    if norm:
        o = (o / l[..., None]).to(V.dtype)
        
    if not head_first:
        o = o.transpose(1, 2)
        rowmax = rowmax.transpose(1, 2)
        l = l.transpose(1, 2)
        
    if norm:
        return o
    else:
        return o, l, rowmax.squeeze(-1).to(torch.float32)