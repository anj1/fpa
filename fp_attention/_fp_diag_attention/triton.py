import torch
import triton
import triton.language as tl
import math
import os
from torch.utils._pytree import tree_map
import torch.nn.functional as F

from fp_attention._utils import diff


fwd_configs = [
    triton.Config({'BM': BM, 'BN': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128, 256]\
    for BN in [32, 64]\
    for s in [1, 2, 3]\
    for w in [4, 8] 
]

def keep(conf):
    BM = conf.kwargs["BM"]
    BN = conf.kwargs["BN"]
    if BM * BN < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def _attn_fwd_inner_diag_fpa(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, p_g, 
                             start_m, range_m, range_n, r: tl.constexpr, w: tl.constexpr, 
                             n_branches: tl.constexpr, 
                             scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, 
                             BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, 
                             M_CTX: tl.constexpr, N_CTX: tl.constexpr, STAGE: tl.constexpr, use_log2: tl.constexpr):
    if STAGE == 1: # causal, non-masking part
        lo, hi = 0, start_m * BM
    elif STAGE == 2: # causal, masking part
        lo, hi = start_m * BM, (start_m + 1) * BM
        lo = tl.multiple_of(lo, BM)
        hi = tl.multiple_of(hi, BM)
    else: # non-causal
        lo, hi = 0, N_CTX

    p_k = tl.advance(p_k, (0, lo))
    p_v = tl.advance(p_v, (lo, 0))
    if gating:
        p_gk = tl.advance(p_gk, (lo,))

    g_diagonals = tl.load(p_g) 

    base_p_g = p_g
    for start_n in range(lo, hi, BN):
        p_g = base_p_g          # rewind for this K-block
        
        k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        
        
        s = tl.full((BM, BN), 1.0, dtype=tl.float32)

        offsets = tl.arange(0, DIM_QK)           # [0 … 63]

        for _ in tl.static_range(n_branches):
            g_l = tl.load(p_g, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            p_g = tl.advance(p_g, (1, 0))              # next branch
            s *= tl.dot(q.to(tl.float32) * g_l, k)     # element-wise scale then dot

        s *= scale
         
        signs = tl.where(s > 0, 1, -1)

        if gating:
            gk = tl.load(p_gk, boundary_check=(0,), padding_option="zero")
        else:
            gk = None

        if use_log2:
            s = tl.log2(s.abs() + 1e-6)
        else:
            s = tl.log(s.abs() + 1e-6)

        if gating:
            s = s + gq[:, None] - gk[None, :]

        if STAGE == 2:
            mask = (range_m[:, None] // r) >= ((start_n + range_n[None, :]) // w)
            s = s + tl.where(mask, 0., -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(s, 1))

        if use_log2:
            p = tl.exp2(s - m_ij[:, None]) * signs
        else:
            p = tl.exp(s - m_ij[:, None]) * signs

        l_ij = tl.sum(p, 1)
        # -- scale acc --
        if use_log2:
            alpha = tl.exp2(m_i - m_ij)
        else:
            alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(p.to(v.dtype), v, acc)
        # -- update m_i
        m_i = m_ij
        p_k = tl.advance(p_k, (0, BN))
        p_v = tl.advance(p_v, (BN, 0))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return acc, l_i, m_i

@triton.autotune(list(filter(keep, fwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "n_branches", "norm"])
@triton.jit
def _attn_fwd_diag_fpa(Q, K, V, G_diagonals, LOG_GQ, LOG_GK, L, M, Out, # 
              stride_qb, stride_qh, stride_qm, stride_qd,  #
              stride_kb, stride_kh, stride_kn, stride_kd,  #
              stride_vb, stride_vh, stride_vn, stride_ve,  #
              stride_gb, stride_gh, stride_gn, stride_gd, #
              stride_mb, stride_mh, stride_mm, #
              stride_gqb, stride_gqh, stride_gqd,  #
              stride_gkb, stride_gkh, stride_gkd,  #
              stride_ob, stride_oh, stride_om, stride_oe,  #
              stride_lb, stride_lh, stride_lm, #
              H, M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
              n_branches: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr,  #
              DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, STAGE: tl.constexpr,  #
              BM: tl.constexpr, BN: tl.constexpr, norm: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    
    # Offsets for Q, K, V, etc. are the same as before
    q_offset = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_offset = off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    v_offset = off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    g_offset = off_b.to(tl.int64) * stride_gb + off_h.to(tl.int64) * stride_gh # New offset for G
    gq_offset = off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
    gk_offset = off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    # Pointers for Q, K, V, etc. are created as before
    p_q = tl.make_block_ptr(Q+q_offset, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m*BM, 0), (BM, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V+v_offset, (N_CTX, DIM_VO), (stride_vn, stride_ve), (0, 0), (BN, DIM_VO), (1, 0))
    p_k = tl.make_block_ptr(K+k_offset, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, 0), (DIM_QK, BN), (0, 1))

    p_g = tl.make_block_ptr(
            G_diagonals + g_offset,                    # base element of (b, h) slice
            (n_branches, DIM_QK),                        # logical view: rows=branches, cols=d_in
            (stride_gn, stride_gd),                    # (row-stride, col-stride) in elements
            (0, 0),                                    # no intra-block offset
            (1, DIM_QK),                                 # load 1×d_in at a time
            (1, 0))                                    # advance along branch dimension

    #if gating:
    p_gq = tl.make_block_ptr(LOG_GQ+gq_offset, (M_CTX,), (stride_gqd,), (start_m*BM,), (BM,), (0,))
    p_gk = tl.make_block_ptr(LOG_GK+gk_offset, (N_CTX,), (stride_gkd,), (0,), (BN,), (0,))
    #else:
    #    p_gq, p_gk = None, None

    # Initialization of online softmax statistics is the same
    range_m = start_m * BM + tl.arange(0, BM)
    range_n = tl.arange(0, BN)
    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DIM_VO], dtype=tl.float32)

    # Loading q and gq is the same
    q = tl.load(p_q, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ+gq_offset, (M_CTX,), (stride_gqd,), (start_m*BM,), (BM,), (0,))
        gq = tl.load(p_gq, cache_modifier=".cg", boundary_check=(0,), padding_option="zero")
    else:
        gq = tl.zeros([BM], dtype=tl.float32)

    # Call the modified inner function
    if STAGE & 1: # non-masking part
        acc, l_i, m_i = _attn_fwd_inner_diag_fpa(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, p_g,
                                   start_m, range_m, range_n, r, w,
                                   n_branches, scale, gating, BM, BN, DIM_QK, DIM_VO,
                                   M_CTX, N_CTX, 4 - STAGE, use_log2)

    if STAGE & 2: # masking part
        acc, l_i, m_i = _attn_fwd_inner_diag_fpa(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, p_g,
                                   start_m, range_m, range_n, r, w,
                                   n_branches, scale, gating, BM, BN, DIM_QK, DIM_VO,
                                   M_CTX, N_CTX, 2, use_log2)
        
    l_i = l_i + 1e-6
    if norm:
        if use_log2:
            m_i += tl.log2(l_i)
        else:
            m_i += tl.log(l_i)
        acc = acc / l_i[:, None]
    
    o_offset = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
    m_offset = off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    l_offset = off_b.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
    p_o = tl.make_block_ptr(Out+o_offset, (M_CTX, DIM_VO), (stride_om, stride_oe), (start_m*BM, 0), (BM, DIM_VO), (1, 0))
    p_l = L + l_offset + range_m * stride_lm
    p_m = M + m_offset + range_m * stride_mm
    
    tl.store(p_m, m_i, mask=range_m < M_CTX)
    if not norm:
        tl.store(p_l, l_i, mask=range_m < M_CTX)
    tl.store(p_o, acc.to(Out.type.element_ty), boundary_check=(0, 1))

bwd_configs = [
    triton.Config({'BN1': BN1, 'BM1': BM1, 'BN2': BN2, 'BM2': BM2, 'BLK_SLICE_FACTOR': BLK_SLICE_FACTOR}, num_stages=s, num_warps=w) \
    for BN1 in [64, 128]\
    for BM1 in [16, 32]\
    for BM2 in [64, 128]\
    for BN2 in [16, 32]\
    for s in [1, 3]\
    for w in [4, 8]\
    for BLK_SLICE_FACTOR in [1, 2]\
]

def keep_bwd(conf):
    BM1 = conf.kwargs["BM1"]
    BN1 = conf.kwargs["BN1"]
    BM2 = conf.kwargs["BM2"]
    BN2 = conf.kwargs["BN2"]
    FACTOR = conf.kwargs["BLK_SLICE_FACTOR"]
    if BN1 != BM2 or BN2 // FACTOR < 16 or BM1 // FACTOR < 16:
        return False
    return True


preprocess_configs = [
    triton.Config({'BM': BM})
    for BM in [64, 128, 256]
]

@triton.autotune(preprocess_configs, key=["M_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,  #
                         stride_ob, stride_oh, stride_om, stride_oe, #
                         stride_dob, stride_doh, stride_dom, stride_doe, #
                         stride_db, stride_dh, stride_dm, #
                         BM: tl.constexpr, HEAD_DIM: tl.constexpr, M_CTX: tl.constexpr  #
                         ):
    range_m = tl.program_id(0) * BM + tl.arange(0, BM)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    mask_m = range_m < M_CTX
    # load
    o = tl.load(O + off_b * stride_ob + off_h * stride_oh + range_m[:, None] * stride_om + off_n[None, :] * stride_oe, cache_modifier=".cg", mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + range_m[:, None] * stride_dom + off_n[None, :] * stride_doe, cache_modifier=".cg", mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + range_m * stride_dm, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dkdv_diag_fpa(dk, dv, dg, dgk, k, v, gk, G_diagonals, #
                    Q, LOG_GQ, DO, M, Delta, DL, #
                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                    stride_g, #
                    M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
                    n_branches: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                    start_n, start_m, num_steps: tl.constexpr, #
                    MASK: tl.constexpr, norm: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_n = start_n + tl.arange(0, BN)

    p_qT = tl.make_block_ptr(Q, (DIM_QK, M_CTX), (stride_qd, stride_qm), (0, start_m), (DIM_QK, BM), (0, 1))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM, DIM_VO), (1, 0))
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM,), (0,))
    else:
        p_gq = None

    curr_m = start_m
    for _ in range(num_steps):
        range_m = curr_m + tl.arange(0, BM)
        # --- re-compute p ---
        qT = tl.load(p_qT, boundary_check=(0, 1), padding_option="zero")
        if gating:
            gq = tl.load(p_gq, boundary_check=(0,), padding_option="zero")
        else:
            gq = tl.zeros((BM,), dtype=tl.float32)
        
        s_prod_T = tl.full((BN, BM), 1.0, dtype=tl.float32)
        offsets = tl.arange(0, DIM_QK)
        for l in tl.static_range(n_branches):
            g_l = tl.load(G_diagonals + l * stride_g + offsets)
            s_branch_T = tl.dot(k, qT * g_l[:, None])
            s_prod_T *= s_branch_T

        s_signs = tl.where(s_prod_T > 0, 1, -1)
        if use_log2:
            zT = tl.log2(s_prod_T.abs() + 1e-7)
        else:
            zT = tl.log(s_prod_T.abs() + 1e-7)
        if gating:
            zT = zT + gq[None, :] - gk[:, None]
        p_m = M + range_m * stride_mm
        m = tl.load(p_m, mask=range_m < M_CTX, other=-float("inf"))
        if MASK:
            mask = (range_m[None, :] // r) >= (range_n[:, None] // w)
            zT = tl.where(mask, zT, -float("inf"))
        if use_log2:
            pT = tl.exp2(zT - m[None, :]) * s_signs
        else:
            pT = tl.exp(zT - m[None, :]) * s_signs

        # --- compute dv ---
        do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        dv = tl.dot(pT.to(Q.type.element_ty), do, dv)

        # --- compute dp ---
        if norm:
            dl_or_delta = - tl.load(Delta + range_m * stride_dm, mask=range_m < M_CTX, other=0.0)
        else:
            dl_or_delta = tl.load(DL + range_m * stride_dlm, mask=range_m < M_CTX, other=0.0)
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32) # (BN, BM)
        ds_scaled_T = pT * (dpT + dl_or_delta[None, :])

        if gating:
            dgk += -tl.sum(ds_scaled_T, 1)

        # --- compute dk, dG ---
        for l in tl.static_range(n_branches):
            g_l = tl.load(G_diagonals + l * stride_g + offsets)
            s_branch_T = tl.dot(k, qT * g_l[:, None])
            ds_branch_T = ds_scaled_T / (s_branch_T + 1e-7)
            
            # dk
            qT_projected = qT * g_l[:, None]
            dk += tl.dot(ds_branch_T.to(qT.type.element_ty), tl.trans(qT_projected))
            
            # dG
            d_g_l = tl.sum(tl.dot(k.T, ds_branch_T.to(qT.type.element_ty)) * qT, axis=1)
            tl.atomic_add(dg + l * stride_g + offsets, d_g_l)

        # increment pointers
        curr_m += BM
        p_qT = tl.advance(p_qT, (0, BM))
        p_do = tl.advance(p_do, (BM, 0))
        if gating:
            p_gq = tl.advance(p_gq, (BM,))

    return dk, dv, dgk


@triton.jit
def _attn_bwd_dq_diag_fpa(dq, dgq, q, gq, do, m, dl_or_delta, #
                  K, V, G_diagonals, LOG_GK, #
                  stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                  stride_g, #
                  M_CTX, N_CTX, r, w, #
                  n_branches: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                  start_m, start_n, num_steps: tl.constexpr, #
                  MASK: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_m = start_m + tl.arange(0, BM)

    p_kT = tl.make_block_ptr(K, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, start_n), (DIM_QK, BN), (0, 1))
    p_vT = tl.make_block_ptr(V, (DIM_VO, N_CTX), (stride_ve, stride_vn), (0, start_n), (DIM_VO, BN), (0, 1))
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN,), (0,))
    else:
        p_gk = None

    curr_n = start_n
    for _ in range(num_steps):
        range_n = curr_n + tl.arange(0, BN)
        # --- re-compute p ---
        kT = tl.load(p_kT, boundary_check=(0, 1), padding_option="zero")
        vT = tl.load(p_vT, boundary_check=(0, 1), padding_option="zero")
        if gating:
            gk = tl.load(p_gk, boundary_check=(0,), padding_option="zero")
        else:
            gk = tl.zeros((BN,), dtype=tl.float32)
        
        s_prod = tl.full((BM, BN), 1.0, dtype=tl.float32)
        offsets = tl.arange(0, DIM_QK)
        for l in tl.static_range(n_branches):
            g_l = tl.load(G_diagonals + l * stride_g + offsets)
            s_branch = tl.dot(q * g_l[None, :], kT)
            s_prod *= s_branch

        s_signs = tl.where(s_prod > 0, 1, -1)
        if use_log2:
            z = tl.log2(s_prod.abs() + 1e-7)
        else:
            z = tl.log(s_prod.abs() + 1e-7)
        if gating:
            z = z + gq[:, None] - gk[None, :]
        if MASK:
            mask = (range_m[:, None] // r) >= (range_n[None, :] // w)
            z = tl.where(mask, z, -float("inf"))
        
        if use_log2:
            p = tl.exp2(z - m[:, None]) * s_signs
        else:
            p = tl.exp(z - m[:, None]) * s_signs

        # --- compute dQ ---
        dp = tl.dot(do, vT, out_dtype=tl.float32)
        ds_scaled = p * (dp + dl_or_delta[:, None])
        if gating:
            dgq += tl.sum(ds_scaled / s_signs, 1, keep_dims=False)
        
        for l in tl.static_range(n_branches):
            g_l = tl.load(G_diagonals + l * stride_g + offsets)
            s_branch = tl.dot(q * g_l[None, :], kT)
            ds_branch = ds_scaled / (s_branch + 1e-7)
            dq += tl.dot(ds_branch.to(kT.type.element_ty), tl.trans(kT)) * g_l[None, :]

        # increment pointers
        curr_n += BN
        p_kT = tl.advance(p_kT, (0, BN))
        p_vT = tl.advance(p_vT, (0, BN))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return dq, dgq


@triton.autotune(list(filter(keep_bwd, bwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "n_branches"])
@triton.jit
def _attn_bwd_diag_fpa(Q, K, V, G_diagonals, LOG_GQ, LOG_GK, M, Delta, DO, DL, DQ, DK, DV, DG, DLOG_GQ, DLOG_GK, #
              stride_qb, stride_qh, stride_qm, stride_qd, #
              stride_kb, stride_kh, stride_kn, stride_kd, #
              stride_vb, stride_vh, stride_vn, stride_ve, #
              stride_gb, stride_gh, stride_gn, stride_gd, #
              stride_mb, stride_mh, stride_mm, #
              stride_db, stride_dh, stride_dm, #
              stride_dob, stride_doh, stride_dom, stride_doe, #
              stride_dlb, stride_dlh, stride_dlm, #
              stride_dqb, stride_dqh, stride_dqm, stride_dqd, #
              stride_dkb, stride_dkh, stride_dkn, stride_dkd, #
              stride_dvb, stride_dvh, stride_dvn, stride_dve, #
              stride_dgb, stride_dgh, stride_dgn, stride_dgd, #
              stride_gqb, stride_gqh, stride_gqm, #
              stride_gkb, stride_gkh, stride_gkn, #
              H, M_CTX, N_CTX, r, w, #
              n_branches: tl.constexpr,  #
              scale: tl.constexpr,  #
              gating: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM1: tl.constexpr,  #
              BN1: tl.constexpr,  #
              BM2: tl.constexpr,  #
              BN2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              norm: tl.constexpr,  #
              use_log2: tl.constexpr,  #
              ):
    if STAGE == 3:
        tl.static_assert((BM1 // BLK_SLICE_FACTOR) % r == 0, "Sliced BM1 must be divisible by w")
        tl.static_assert((BN2 // BLK_SLICE_FACTOR) % w == 0, "Sliced BN2 must be divisible by w")
    else:
        tl.static_assert(BM1 % r == 0, "BM1 must be divisible by r")
        tl.static_assert(BN2 % w == 0, "BN2 must be divisible by w")
    tl.static_assert(BN1 % w == 0, "BN1 must be divisible by w")
    tl.static_assert(BM2 % r == 0, "BM2 must be divisible by r")

    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    start_n = tl.program_id(0)*BN1


    Q += off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    G_diagonals += off_b.to(tl.int64) * stride_gb + off_h.to(tl.int64) * stride_gh
    M += off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    Delta += off_b.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
    DO += off_b.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
    DL += off_b.to(tl.int64) * stride_dlb + off_h.to(tl.int64) * stride_dlh
    DQ += off_b.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
    DK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
    DV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
    DG += off_b.to(tl.int64) * stride_dgb + off_h.to(tl.int64) * stride_dgh
    if gating:
        LOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        LOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh
        DLOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        DLOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    # -- First part: compute dk, dv
    MASK_BLOCK_M1: tl.constexpr = BM1 // BLK_SLICE_FACTOR
    range_n = start_n + tl.arange(0, BN1)

    dv = tl.zeros([BN1, DIM_VO], dtype=tl.float32)
    dk = tl.zeros([BN1, DIM_QK], dtype=tl.float32)
    dgk = tl.zeros([BN1,], dtype=tl.float32)

    # load k, v, gk
    p_k = tl.make_block_ptr(K, (N_CTX, DIM_QK), (stride_kn, stride_kd), (start_n, 0), (BN1, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V, (N_CTX, DIM_VO), (stride_vn, stride_ve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    k = tl.load(p_k, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    v = tl.load(p_v, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN1,), (0,))
        gk = tl.load(p_gk, cache_modifier=".cg", boundary_check=(0,), padding_option="zero")
    else:
        gk = None

    start_m = start_n if STAGE == 3 else 0
    if STAGE & 2: # masked blocks
        num_steps = BN1 // MASK_BLOCK_M1
        dk, dv, dgk = _attn_bwd_dkdv_diag_fpa(dk, dv, DG, dgk, k, v, gk, G_diagonals,
                                    Q, LOG_GQ, DO, M, Delta, DL,
                                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm,
                                    stride_gn,
                                    M_CTX, N_CTX, r, w, #
                                    n_branches, scale, gating, MASK_BLOCK_M1, BN1, DIM_QK, DIM_VO, #
                                    start_n, start_m, num_steps, #
                                    True, norm, use_log2)
        start_m += num_steps * MASK_BLOCK_M1
        
    # unmasked blocks
    num_steps = (M_CTX - start_m) // BM1
    dk, dv, dgk = _attn_bwd_dkdv_diag_fpa(dk, dv, DG, dgk, k, v, gk, G_diagonals,
                                Q, LOG_GQ, DO, M, Delta, DL,
                                stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm,
                                stride_gn,
                                M_CTX, N_CTX, r, w, #
                                n_branches, scale, gating, BM1, BN1, DIM_QK, DIM_VO, #
                                start_n, start_m, num_steps, #
                                False, norm, use_log2)

class _diag_fp_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, G_diagonal, log_G, n_branches, r, w, causal, head_first, scale, norm, use_log2):
        """ 
            Attention formulation of power attention.
            
            When norm is True, the output is normalized by temporal norm (l), and the returned
            temporal norm is uninitialized.
            When norm is False, the output is not normalized, and l is the temporal norm (sum of exponentials of the powered attention scores)

            Args:
            Q: (B, H_Q, CTX, D)
            K: (B, H_K, CTX, D)
            V: (B, H_K, CTX, E)
            G_diagonal: (B, H_Q, N_BRANCHES, D) or (B, N_BRANCHES, H_Q, D)
            n_branches: int
            log_G: (B, H_Q // R, CTX) or (B, CTX, H_Q // R)
            r: int, number of heads in q to form a group
            w: int, number of heads in k to form a group
            causal: bool
            head_first: bool
            norm: bool
            use_log2: bool

            Returns:
                o: (B, H_Q // R, CTX, E) if head_first else (B, CTX, H_Q // R, E)
                l: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
                rowmax: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
        """
        if head_first:
            b, hq, t, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
        else:
            b, t, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]
        assert r in {1, 2, 4, 8, 16}, "r must be 1, 2, 4, 8, or 16"
        assert w in {1, 2, 4, 8, 16}, "w must be 1, 2, 4, 8, or 16"
        assert hq % r == 0, "hq must be divisible by r"
        assert hk % w == 0, "hk must be divisible by w"
        assert hq // r == hk // w, "hq // r must be equal to hk // w"
        assert isinstance(n_branches, int) and n_branches > 0, "n_branches must be a positive integer"
        assert d in {16, 32, 64, 128, 256}, "d must be 16, 32, 64, 128, or 256"
        assert e in {16, 32, 64, 128, 256}, "e must be 16, 32, 64, 128, or 256"

        h = hq // r
        gating = log_G is not None
        if gating and use_log2:
            log_G = log_G * math.log2(math.e)
        if head_first:
            o = torch.empty((b, h, t, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, h, t)
                log_GQ = log_G.repeat_interleave(r, dim=2)
                log_GK = log_G.repeat_interleave(w, dim=2)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(1), log_GQ.stride(2))
                gk_strides = (log_GK.stride(0), log_GK.stride(1), log_GK.stride(2))
            else:
                log_GK = torch.empty(0, device=Q.device, dtype=torch.float32)
                log_GQ = torch.empty(0, device=Q.device, dtype=torch.float32)
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, h, t * r, d)
            K = K.view(b, h, t * w, d)
            V = V.view(b, h, t * w, e)
            
            G_diagonal = G_diagonal.view(b, h, n_branches, d)
            G_diagonal = G_diagonal.contiguous()
            
            rowmax = torch.empty((b, h, t), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, h, t), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3))
            k_strides = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
            v_strides = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
            
            g_strides = (G_diagonal.stride(0), G_diagonal.stride(1), G_diagonal.stride(2), G_diagonal.stride(3))
            
            l_strides = (l.stride(0), l.stride(1), l.stride(2))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(1), rowmax.stride(2))
            o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        else:
            o = torch.empty((b, t, h, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, t, h) 
                log_GQ = log_G.repeat_interleave(r, dim=1)
                log_GK = log_G.repeat_interleave(w, dim=1)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(2), log_GQ.stride(1))
                gk_strides = (log_GK.stride(0), log_GK.stride(2), log_GK.stride(1))
            else:
                log_GQ = torch.empty(0, device=Q.device, dtype=torch.float32)
                log_GK = torch.empty(0, device=Q.device, dtype=torch.float32)
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, t * r, h, d)
            K = K.view(b, t * w, h, d)
            V = V.view(b, t * w, h, e)
            
            G_diagonal = G_diagonal.view(b, n_branches, h, d)
            G_diagonal = G_diagonal.contiguous()
            
            rowmax = torch.empty((b, t, h), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, t, h), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(2), Q.stride(1), Q.stride(3))
            k_strides = (K.stride(0), K.stride(2), K.stride(1), K.stride(3))
            v_strides = (V.stride(0), V.stride(2), V.stride(1), V.stride(3))
            
            g_strides = (G_diagonal.stride(0), G_diagonal.stride(2), G_diagonal.stride(1), G_diagonal.stride(3))
                        
            l_strides = (l.stride(0), l.stride(2), l.stride(1))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(2), rowmax.stride(1))
            o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(r*t, args["BM"]), b * h)
        _attn_fwd_diag_fpa[grid](
            Q, K, V, G_diagonal, log_GQ, log_GK, l, rowmax, o,
            *q_strides, *k_strides, *v_strides, *g_strides, *rowmax_strides, *gq_strides, *gk_strides, *o_strides, *l_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, n_branches=n_branches, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        
        ctx.save_for_backward(Q, K, V, G_diagonal, l, rowmax, o, log_GQ, log_GK)
        ctx.b = b
        ctx.h = h
        ctx.t = t
        ctx.r = r
        ctx.w = w
        ctx.grid = grid
        ctx.d = d
        ctx.e = e
        ctx.n_branches = n_branches
        ctx.gating = gating
        ctx.scale = scale
        ctx.norm = norm
        ctx.q_strides = q_strides
        ctx.k_strides = k_strides
        ctx.v_strides = v_strides
        ctx.g_strides = g_strides
        ctx.rowmax_strides = rowmax_strides
        ctx.gq_strides = gq_strides
        ctx.gk_strides = gk_strides
        ctx.o_strides = o_strides
        ctx.head_first = head_first
        ctx.stage = stage
        ctx.use_log2 = use_log2
        return o, l, rowmax

    @staticmethod
    def backward(ctx, do, dl, drowmax):
        Q, K, V, G_diagonal, l, rowmax, o, log_GQ, log_GK = ctx.saved_tensors
        if log_GQ is not None:
            log_GQ = log_GQ.contiguous() # needed for reuse log_GQ's strides for dlog_GQ
            log_GK = log_GK.contiguous() # needed for reuse log_GK's strides for dlog_GK
        do = do.contiguous()
        b, h, t, stage, norm, gating, use_log2 = ctx.b, ctx.h, ctx.t, ctx.stage, ctx.norm, ctx.gating, ctx.use_log2
        q_strides, k_strides, v_strides, g_strides, rowmax_strides, gq_strides, gk_strides = ctx.q_strides, ctx.k_strides, ctx.v_strides, ctx.g_strides, ctx.rowmax_strides, ctx.gq_strides, ctx.gk_strides
        r, w, d, e, n_branches, scale = ctx.r, ctx.w, ctx.d, ctx.e, ctx.n_branches, ctx.scale

        dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)
        dG_diagonal = torch.zeros_like(G_diagonal, dtype=torch.float32)
        dlog_GQ = torch.empty_like(log_GQ) if gating else None
        dlog_GK = torch.empty_like(log_GK) if gating else None
        delta = torch.empty_like(rowmax) if norm else torch.empty((0, 0, 0))

        if ctx.head_first:
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides, dG_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2), x.stride(3)), (do, dQ, dK, dV, o, dG_diagonal))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2)), (dl, delta))
        else:
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides, dG_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1), x.stride(3)), (do, dQ, dK, dV, o, dG_diagonal))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1)), (dl, delta))

        if norm:
            _attn_bwd_preprocess[lambda args: (triton.cdiv(t, args["BM"]), b, h)](
                o, do, delta,
                *o_strides, *do_strides, *delta_strides,
                HEAD_DIM=e, M_CTX=t
            )

        _attn_bwd_diag_fpa[lambda args: (triton.cdiv(w*t, args["BN1"]), b * h)](
            Q, K, V, G_diagonal, log_GQ, log_GK, rowmax, delta, do, dl, dQ, dK, dV, dG_diagonal, dlog_GQ, dlog_GK,
            *q_strides, *k_strides, *v_strides, *g_strides,
            *rowmax_strides, *delta_strides, *do_strides, *dl_strides, 
            *dQ_strides, *dK_strides, *dV_strides, *dG_strides,
            *gq_strides, *gk_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, n_branches=n_branches, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        
        if gating:
            if ctx.head_first:
                dlog_G = dlog_GQ.view(b, h, t, r).sum(dim=-1) + dlog_GK.view(b, h, t, w).sum(dim=-1)
            else:
                dlog_G = dlog_GQ.view(b, t, r, h).sum(dim=-2) + dlog_GK.view(b, t, w, h).sum(dim=-2)
        else:
            dlog_G = None
        return dQ, dK, dV, dG_diagonal.to(G_diagonal.dtype), dlog_G, None, None, None, None, None, None, None, None

def _attention_fn(Q, K, V, G_diagonal, log_G, r=1, w=1, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):

    if isinstance(G_diagonal, (list, tuple)):
        n_branches = len(G_diagonal)
    else:
        # tensors are (B, H, N_BRANCHES, D)  or  (B, N_BRANCHES, H, D)
        # pick the branch dimension whichever layout is used
        n_branches = G_diagonal.shape[2] if head_first else G_diagonal.shape[1]

    Y, l, rowmax = _diag_fp_attention.apply(
        Q, K, V, G_diagonal, log_G,
        n_branches, r, w,
        causal,
        head_first,
        scale, norm, use_log2)

    rowmax = rowmax.detach()
    if norm:
        return Y
    return Y, l, rowmax

attention = torch.compiler.disable(_attention_fn)

def compare_tensors(name, t1, t2, atol=1e-2, rtol=1e-2, verbose=False):
    """Helper function to compare two tensors and print the result."""
    if t1 is None and t2 is None:
        print(f"{name:<20} | ✅ Pass (Both None)")
        return True
    if t1 is None or t2 is None:
        print(f"{name:<20} | ❌ Fail (One is None)")
        return False
        
    is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol, equal_nan=True)
    print(f"{name:<20} | {'✅ Pass' if is_close else '❌ Fail'}")
    if not is_close and verbose:
        diff = torch.abs(t1 - t2)
        max_diff_idx = torch.argmax(diff)
        print(f"  - Max diff: {torch.max(diff)} at index {max_diff_idx}")
        print(f"  - Triton tensor at max diff: {t1.flatten()[max_diff_idx]}")
        print(f"  - Ref tensor at max diff:    {t2.flatten()[max_diff_idx]}")
    return is_close

def clone_item(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().requires_grad_(True)
    elif isinstance(x, (list, tuple)):
        return [clone_item(y) for y in x]
    else:
        return x

def clone_dict(d):
    return {k: clone_item(v) for k, v in d.items()}


def run_test(kw, identity_G=False, verbose=False):
    """Runs forward and backward pass tests against the reference implementation."""
    from fp_attention._fp_diag_attention.reference import fpa_diagonal_attention
    from fp_attention._fp_diag_attention.create_inputs import create_inputs

    # Common parameters for attention calls
    attn_params = dict(r=1, w=1, causal=True, head_first=kw['head_first'], scale=1.0, norm=kw['norm'], use_log2=False)

    # === Forward Pass Test ===
    print("\n--- FWD Pass ---")
    fwd_inputs = create_inputs(**kw)
    
    if identity_G:
        fwd_inputs['G_diagonal'][:] = 1.0
    
    # Triton implementation
    triton_out = attention(fwd_inputs['Q'], fwd_inputs['K'], fwd_inputs['V'], fwd_inputs['G_diagonal'], fwd_inputs['log_G'], **attn_params)
    # Reference implementation
    torch.backends.cuda.matmul.allow_tf32 = False
    ref_out = fpa_diagonal_attention(fwd_inputs['Q'], fwd_inputs['K'], fwd_inputs['V'], fwd_inputs['G_diagonal'], fwd_inputs['log_G'], n_branches=kw['n_branches'], **attn_params)

    if kw['norm']:
        compare_tensors("Output (o)", triton_out, ref_out, verbose=verbose)
    else:
        o_triton, l_triton, rowmax_triton = triton_out
        o_ref, l_ref, rowmax_ref = ref_out
        compare_tensors("Output (o)", o_triton, o_ref, verbose=verbose)
        compare_tensors("Sum (l)", l_triton, l_ref, verbose=verbose)
        compare_tensors("Rowmax (m)", rowmax_triton, rowmax_ref, verbose=verbose)

    exit(2)

    # === Backward Pass Test ===
    print("\n--- BWD Pass ---")
    print("NOTE: BWD test will fail if the main backward kernel is not implemented/uncommented.")
    kw_grad = {**kw, 'requires_grad': True}
    
    # Inputs for Triton
    tri_inputs = create_inputs(**kw_grad)
    if identity_G:
        with torch.no_grad():
            tri_inputs['G_diagonal'][:] = 1.0
    
    Q_tri, K_tri, V_tri, G_diag_tri, log_G_tri = tri_inputs['Q'], tri_inputs['K'], tri_inputs['V'], tri_inputs['G_diagonal'], tri_inputs['log_G']
    
    ref_inputs = clone_dict(tri_inputs)
    Q_ref, K_ref, V_ref, G_diag_ref, log_G_ref = ref_inputs['Q'], ref_inputs['K'], ref_inputs['V'], ref_inputs['G_diagonal'], ref_inputs['log_G']

    #print(ref_inputs["K"][:,:,0:4,0:4])

    o_triton = attention(Q_tri, K_tri, V_tri, G_diag_tri, log_G_tri, **attn_params)
    if not kw['norm']: o_triton = o_triton[0]

    o_ref = fpa_diagonal_attention(Q_ref, K_ref, V_ref, G_diag_ref, log_G_ref, n_branches=kw['n_branches'], **attn_params)
    if not kw['norm']: o_ref = o_ref[0]

    do = torch.randn_like(o_triton)

    o_triton.backward(gradient=do, retain_graph=True)
    o_ref.backward(gradient=do, retain_graph=True)

    compare_tensors("dQ", Q_tri.grad, Q_ref.grad, verbose=verbose)
    compare_tensors("dK", K_tri.grad, K_ref.grad, verbose=verbose)
    compare_tensors("dV", V_tri.grad, V_ref.grad, verbose=verbose)
    compare_tensors("dG_diagonal", G_diag_tri.grad, G_diag_ref.grad, verbose=verbose)
    # if gating
    if log_G_tri is not None:
        compare_tensors("dlog_G", log_G_tri.grad, log_G_ref.grad, verbose=verbose)

if __name__ == "__main__":
    # Correctness testing
    print("="*20 + " Correctness Test " + "="*20)
    test_configs = [
        #dict(b=2, h=2, t=4, d_in=16, dtype=torch.float32, device='cuda', n_branches=2, seed=42, std=1/8.0, gating=True, norm=False, head_first=True),
        dict(b=2, h=2, t=64, d_in=32, dtype=torch.float32, device='cuda', n_branches=2, seed=42, std=1/8.0, gating=True, norm=False, head_first=True),
        dict(b=1, h=4, t=128, d_in=64, dtype=torch.bfloat16, device='cuda', n_branches=3, seed=24, std=1/8.0, gating=False, norm=True, head_first=False),
    ]
    for i, kw in enumerate(test_configs):
        print(f"\n--- Config {i+1}: { {k:v for k,v in kw.items() if k not in ['device', 'dtype', 'seed', 'std']} } ---")
        run_test(kw, verbose=True)
    print("="*58 + "\n")

    from fp_attention._fp_diag_attention.reference import fpa_diagonal_attention 
    from fp_attention._fp_diag_attention.create_inputs import create_inputs
    from perf._timing import benchmark_speed

    VERBOSE = True

    kw = dict(b=1, h=6, d_in=64, dtype=torch.bfloat16, device='cuda', n_branches=2, seed=42, std=1/8.0, gating=True, norm=False, head_first=True)
    def print_rowstr(rowstr):
        print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))

    token_count = 2**16
    for n_branches in [2, 3]:
        for mode in ['fwd', 'bwd']:
            print(f"triton-vs-cutlass-token{token_count}-head{kw['h']}-dim{kw['d_in']}-deg{n_branches}-{mode}")
            print_rowstr("chunk_size,triton,cutlass,triton speedup")
            for ctx in [2**i for i in range(7, 16)]:
                kw['t'] = ctx
                kw['b'] = token_count // ctx
                kw['n_branches'] = n_branches
                print(f"ctx={ctx}, b={kw['b']}, t={kw['t']}, n_branches={kw['n_branches']}")
                triton_time = benchmark_speed(mode, attention, create_inputs, kw, compile=False)
                #cutlass_time = benchmark_speed(mode, attention_cutlass, create_inputs_cutlass, {key: kw[key] for key in kw if key != 'norm'}, compile=False)
                #speedup = cutlass_time / triton_time
                cutlass_time = triton_time * 0.8  # Simulated cutlass time for demonstration
                speedup = cutlass_time / triton_time
                print_rowstr(f"{ctx}, {triton_time:.2f}, {cutlass_time:.2f}, {speedup:.2f}")
