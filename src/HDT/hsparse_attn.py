"""
Hierarchical Attention
===============

This is a Triton implementation based on the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import math
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, hash_ids0, hash_ids1, hash_ids2,
                keep_ids0, keep_ids1, keep_ids2, anchor_pos,
                sm_scale,  L,  Out, A, output_attentions: tl.constexpr, #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                stride_az, stride_ah, stride_am, stride_an,
                stride_zid,
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                IS_CAUSAL: tl.constexpr  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, vk_offset),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(vk_offset, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e6
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)

    offs_hash = (off_hz // H) * stride_zid + offs_m
    qh_vals = tl.load(hash_ids0 + offs_hash, mask=offs_m < N_CTX, other=1e9)
    qi = tl.load(keep_ids0 + offs_hash, mask=offs_m < N_CTX, other=0)
    qi1 = tl.load(keep_ids1 + offs_hash, mask=offs_m < N_CTX, other=0)
    min_q_hash = tl.min(qh_vals, axis=0)
    qh_vals = tl.where(qh_vals == 1e9, -1, qh_vals)
    max_q_hash = tl.max(qh_vals, axis=0)
    min_kh = -1
    lo = 0
    hi = 0
    offs_kh = (off_hz // H) * stride_zid + offs_n
    if min_q_hash != 1e9:
        # # Increment the start and end to find start and end blocks
        while min_kh <= max_q_hash:
            kh_vals = tl.load(hash_ids0 + offs_kh, mask=offs_n < N_CTX, other=+1e9)
            min_kh = tl.min(kh_vals, axis=0)
            if min_kh <= max_q_hash and min_kh != 1e9:
                hi += 1
            kh_vals = tl.where(kh_vals == 1e9, -1e9, kh_vals)
            max_kh = tl.max(kh_vals, axis=0)
            if max_kh < min_q_hash and max_kh != -1e9:
                lo += 1
            offs_n += BLOCK_N
            offs_kh += BLOCK_N
        K_block_ptr = tl.advance(K_block_ptr, (0, lo * BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (lo * BLOCK_N, 0))
        offs_n = BLOCK_N * lo + tl.arange(0, BLOCK_N)  # indices of keys we want to process, we start from [0, BLOCK_N-1] and update in the loop
        offs_kid = (off_hz // H) * stride_zid + offs_n
        for start_n in range(lo, hi):
            # Load values for K and K_idx
            kh_vals = tl.load(hash_ids0 + offs_kid, mask=offs_n < N_CTX, other=-1e9)
            kh_vals = tl.where(kh_vals == 1e9, -1e9, kh_vals)
            ki = tl.load(keep_ids0 + offs_kid, mask=offs_n < N_CTX, other=0)
            ki1 = tl.load(keep_ids1 + offs_kid, mask=offs_n < N_CTX, other=0)
            cls_mask = (qi == 1 and qi1 == 1)[:, None] and (ki == 1 and ki1 == 1)[None, :]
            attn_mask = cls_mask == 0 and ((qi[:, None] == 1 and ki[None, :] == 1) and (qh_vals[:, None] == kh_vals[None, :]))
            # -- load k, v --
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
            qk = tl.where(attn_mask, qk, float("-inf"))
            qk += tl.dot(q, k, allow_tf32=True)
            # -- compute scaling constant ---
            # -- update m_i and l_i
            # can't m_i - m_ij because it causes to nan
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            if output_attentions:
                tl.store(A + off_hz * stride_ah + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, qk, mask=offs_n[None, :] < N_CTX)
            # -- scale and update acc --
            acc *= alpha[:, None]
            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            # update pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            offs_kid += BLOCK_N
            offs_n += BLOCK_N
        anchor_offsets = (off_hz // H) * stride_zid + tl.arange(0, BLOCK_N)
        anchor_tokens = tl.load(anchor_pos + anchor_offsets)
        offs_hash = (off_hz // H) * stride_zid + offs_m
        qh1_vals = tl.load(hash_ids1 + offs_hash, mask=offs_m < N_CTX, other=1e9)
        qh2_vals = tl.load(hash_ids2 + offs_hash, mask=offs_m < N_CTX, other=1e9)
        qi1_vals = tl.load(keep_ids1 + offs_hash, mask=offs_m < N_CTX, other=0)
        qi2_vals = tl.load(keep_ids2 + offs_hash, mask=offs_m < N_CTX, other=0)
        while tl.max(anchor_tokens) > -1:
            # anchor_tokens = tl.where(anchor_tokens == -1, 0, anchor_offsets)
            k = tl.load(K + qvk_offset + anchor_tokens[None, :] * stride_kn + offs_k[:, None] * stride_kk, mask=(anchor_tokens != -1)[None, :], other=0)
            v = tl.load(V + qvk_offset + anchor_tokens[:, None] * stride_vn + offs_k[None, :] * stride_vk, mask=(anchor_tokens != -1)[:, None], other=0)
            kh1_vals = tl.load(hash_ids1 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=-1)
            kh2_vals = tl.load(hash_ids2 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=-1)
            ki1_vals = tl.load(keep_ids1 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=0)
            ki2_vals = tl.load(keep_ids2 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=0)
            attn_mask = (((qi1_vals[:, None] == 1 and ki1_vals[None, :] == 1) and (qh1_vals[:, None] == kh1_vals[None, :]))
                    or ((qi2_vals[:, None] == 1 and ki2_vals[None, :] == 1) and (qh2_vals[:, None] == kh2_vals[None, :])))
            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.where(attn_mask, qk, float("-inf"))
            qk += tl.dot(q, k, allow_tf32=True)
            # -- compute scaling constant ---
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            if output_attentions:
                tl.store(A + off_hz * stride_ah + offs_m[:, None] * stride_am + anchor_tokens[None, :] * stride_an, qk, mask=anchor_tokens[None, :] != -1)
            # -- scale and update acc --
            acc *= alpha[:, None]
            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            anchor_offsets += BLOCK_N
            anchor_tokens = tl.load(anchor_pos + anchor_offsets)
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    # O_block_ptr = tl.make_block_ptr(
    #     base=Out,
    #     shape=(Z_H_N_CTX, BLOCK_DMODEL),
    #     strides=(stride_om, stride_on),
    #     offsets=(vk_offset + start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0),
    # )
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel_one_col_block(Q, K, V, hash_ids0, hash_ids1, hash_ids2,
                              keep_ids0, keep_ids1, keep_ids2, anchor_pos,
                              sm_scale, qk_scale,  #
                              Out, DO,  #
                              DQ, DK, DV,  #
                              L,  D,  #
                              Q_block_ptr, K_block_ptr, V_block_ptr,  #
                              DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                              stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                              stride_kz, stride_kh, stride_kn, stride_kk,  #
                              stride_vz, stride_vh, stride_vn, stride_vk,  #
                              stride_zid,
                              Z, H, N_CTX,  #
                              off_h, off_z, off_hz, start_n, num_block,  #
                              BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                              BLOCK_N: tl.constexpr,  #
                              SEQUENCE_PARALLEL: tl.constexpr,  #
                              CAUSAL: tl.constexpr,  #
                              MMA_V3: tl.constexpr  #
                              ):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0
    hi = lo
    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    offs_hash = (off_hz // H) * stride_zid + offs_n
    kh_vals = tl.load(hash_ids0 + offs_hash, mask=offs_n < N_CTX, other=1e9)
    ki = tl.load(keep_ids0 + offs_hash, mask=offs_n < N_CTX, other=0)
    ki1 = tl.load(keep_ids1 + offs_hash, mask=offs_n < N_CTX, other=0)
    min_k_hash = tl.min(kh_vals, axis=0)
    kh_vals = tl.where(kh_vals == 1e9, -1, kh_vals)
    max_k_hash = tl.max(kh_vals, axis=0)
    min_qh = -1
    offs_qh = (off_hz // H) * stride_zid + offs_m
    # # Increment the start and end to find start and end blocks
    while min_qh <= max_k_hash:
        qh_vals = tl.load(hash_ids0 + offs_qh, mask=offs_m < N_CTX, other=+1e9)
        min_qh = tl.min(qh_vals, axis=0)
        if min_qh <= max_k_hash and min_qh != 1e9:
            hi += 1
        qh_vals = tl.where(qh_vals == 1e9, -1e9, qh_vals)
        max_qh = tl.max(qh_vals, axis=0)
        if max_qh < min_k_hash and max_qh != -1e9:
            lo += 1
        offs_m += BLOCK_M
        offs_qh += BLOCK_M

    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa.to(tl.int64) * start_n
    DQ_offset = DQ_offset // stride_qm

    Q_block_ptr = tl.advance(Q_block_ptr, (lo * BLOCK_M + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_N + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_N + V_offset, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo * BLOCK_M + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo * BLOCK_M + DQ_offset, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_N + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_N + V_offset, 0))



    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    offs_m = BLOCK_M * lo + tl.arange(0,
                                      BLOCK_M)  # indices of keys we want to process, we start from [0, BLOCK_N-1] and update in the loop
    offs_qid = (off_hz // H) * stride_zid + offs_m
    # loop over rows
    for start_m in range(lo, hi):
        # Load values for Q and Q_idx
        qh_vals = tl.load(hash_ids0 + offs_qid, mask=offs_m < N_CTX, other=-1e9)
        qh_vals = tl.where(qh_vals == 1e9, -1e9, qh_vals)
        qi = tl.load(keep_ids0 + offs_qid, mask=offs_m < N_CTX, other=0)
        qi1 = tl.load(keep_ids1 + offs_qid, mask=offs_m < N_CTX, other=0)
        cls_mask = (qi == 1 and qi1 == 1)[:, None] and (ki == 1 and ki1 == 1)[None, :]
        attn_mask = cls_mask == 0 and (
                    (qi[:, None] == 1 and ki[None, :] == 1) and (qh_vals[:, None] == kh_vals[None, :]))
        offs_m_curr = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        # load q, k, v, do on-chip
        q = tl.load(Q_block_ptr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.0), float("-inf"))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(attn_mask, qk, float("-inf"))
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        # compute dq
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k, allow_tf32=True)
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k, allow_tf32=True)
            else:
                # not work with mma v3, becuase M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds), allow_tf32=True))
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        offs_qid += BLOCK_M
        offs_m += BLOCK_M

    anchor_offsets = (off_hz // H) * stride_zid + tl.arange(0, BLOCK_M)
    anchor_tokens = tl.load(anchor_pos + anchor_offsets)
    offs_hash = (off_hz // H) * stride_zid + offs_n
    kh1_vals = tl.load(hash_ids1 + offs_hash, mask=offs_n < N_CTX, other=1e9)
    kh2_vals = tl.load(hash_ids2 + offs_hash, mask=offs_n < N_CTX, other=1e9)
    ki1_vals = tl.load(keep_ids1 + offs_hash, mask=offs_n < N_CTX, other=0)
    ki2_vals = tl.load(keep_ids2 + offs_hash, mask=offs_n < N_CTX, other=0)
    qvk_offset = off_hz * stride_qh
    offs_k = tl.arange(0, BLOCK_DMODEL)
    while tl.max(anchor_tokens) > -1:
        # anchor_tokens = tl.where(anchor_tokens == -1, 0, anchor_offsets)
        q = tl.load(Q + qvk_offset + anchor_tokens[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(anchor_tokens != -1)[:, None], other=0)
        qh1_vals = tl.load(hash_ids1 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=-1)
        qh2_vals = tl.load(hash_ids2 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=-1)
        qi1_vals = tl.load(keep_ids1 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=0)
        qi2_vals = tl.load(keep_ids2 + (off_hz // H) * stride_zid + anchor_tokens, mask=anchor_tokens != -1, other=0)
        attn_mask = (((qi1_vals[:, None] == 1 and ki1_vals[None, :] == 1) and (qh1_vals[:, None] == kh1_vals[None, :]))
                     or ((qi2_vals[:, None] == 1 and ki2_vals[None, :] == 1) and (
                            qh2_vals[:, None] == kh2_vals[None, :])))
        # -- compute qk ---
        if CAUSAL:
            qk = tl.where(anchor_tokens[:, None] >= (offs_n[None, :]), float(0.0), float("-inf"))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(attn_mask, qk, float("-inf"))
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + anchor_tokens, mask=anchor_tokens != -1, other=0)
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        do = tl.load(DO + qvk_offset + anchor_tokens[:, None] * stride_qm + offs_k[None, :] * stride_kk, mask=(anchor_tokens != -1)[:, None], other=0)
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + anchor_tokens, mask=anchor_tokens != -1, other=0)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        # compute dq
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ + qvk_offset + anchor_tokens[:, None] * stride_qm + offs_k[None, :] * stride_kk, mask=(anchor_tokens != -1)[:, None], other=0)
            dq += tl.dot(ds, k, allow_tf32=True)
            tl.store(DQ + qvk_offset + anchor_tokens[:, None] * stride_qm + offs_k[None, :] * stride_kk, dq.to(Q.dtype.element_ty), mask=(anchor_tokens != -1)[:, None])

        # increment pointers
        anchor_offsets += BLOCK_M
        anchor_tokens = tl.load(anchor_pos + anchor_offsets)

    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))


@triton.jit
def _bwd_kernel(Q, K, V, hash_ids0, hash_ids1, hash_ids2,
                keep_ids0, keep_ids1, keep_ids2, anchor_pos,
                sm_scale, Out, DO,  #
                DQ, DK, DV,  #
                L,  #
                D,  #
                stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                stride_idz,
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                SQ_Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                SEQUENCE_PARALLEL: tl.constexpr,  #
                CAUSAL: tl.constexpr,  #
                MMA_V3: tl.constexpr  #
                ):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    else:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, hash_ids0, hash_ids1, hash_ids2,
                                      keep_ids0, keep_ids1, keep_ids2, anchor_pos, sm_scale,
                                      qk_scale, Out, DO,  #
                                      DQ, DK, DV,  #
                                      L,  #
                                      D,  #
                                      Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                      DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                      stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                      stride_kz, stride_kh, stride_kn, stride_kk,  #
                                      stride_vz, stride_vh, stride_vn, stride_vk,  #
                                      stride_idz,
                                      Z, H, N_CTX,  #
                                      off_h, off_z, off_hz, start_n, num_block_n,  #
                                      BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                      BLOCK_N=BLOCK_N,  #
                                      SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                      CAUSAL=CAUSAL,  #
                                      MMA_V3=MMA_V3  #
                                      )
    else:
        raise NotImplemented("sequence parallel is not implemented yet")
        # start_n = tl.program_id(1)
        # _bwd_kernel_one_col_block(Q, K, V, hash_ids0, hash_ids1, hash_ids2,
        #                           keep_ids0, keep_ids1, keep_ids2, anchor_pos, sm_scale,
        #                           qk_scale, Out, DO,  #
        #                           DQ, DK, DV,  #
        #                           L,  #
        #                           D,  #
        #                           Q_block_ptr, K_block_ptr, V_block_ptr,  #
        #                           DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
        #                           stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
        #                           stride_kz, stride_kh, stride_kn, stride_kk,  #
        #                           stride_vz, stride_vh, stride_vn, stride_vk,  #
        #                           stride_idz,
        #                           Z, H, N_CTX,  #
        #                           off_h, off_z, off_hz, start_n, num_block_n,  #
        #                           BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
        #                           BLOCK_N=BLOCK_N,  #
        #                           SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
        #                           CAUSAL=CAUSAL,  #
        #                           MMA_V3=MMA_V3  #
        #                           )


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, hash_ids, keep_ids, sm_scale, causal=False, output_attentions=False):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_M = 128
        BLOCK_N = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        anchor_ids = keep_ids[1] + keep_ids[2]
        anchor_pos = torch.nested.to_padded_tensor(torch.nested.nested_tensor([anchor_ids[i].nonzero()[:, 0] for i in range(len(anchor_ids))]), padding=-1, output_size=(q.shape[0], q.shape[2]))
        if output_attentions:
            A = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[2]), device=q.device, dtype=torch.float32) - 1e9
            stride_az, stride_ah, stride_am, stride_an = A.stride(0), A.stride(1), A.stride(2), A.stride(3)
        else:
            A = hash_ids[0] # no meaning, just a placeholder
            stride_az, stride_ah, stride_am, stride_an = 0, 0, 0, 0
        _fwd_kernel[grid](
            q, k, v, hash_ids[0], hash_ids[1], hash_ids[2],
            keep_ids[0], keep_ids[1], keep_ids[2], anchor_pos,
            sm_scale,  L,  o,  A, output_attentions, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            stride_az, stride_ah, stride_am, stride_an,
            hash_ids[0].stride(0),
            q.shape[0], q.shape[1], q.shape[2],  #
            q.shape[0] * q.shape[1] * q.shape[2],  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,  #
            IS_CAUSAL=causal,  #
            num_warps=num_warps,  #
            num_stages=4  #
        )
        ctx.mark_non_differentiable(A)
        ctx.save_for_backward(q, k, v, o, L, hash_ids[0], hash_ids[1], hash_ids[2], keep_ids[0], keep_ids[1], keep_ids[2], anchor_pos)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = False
        return o, torch.softmax(A, dim=-1).detach().cpu() if output_attentions else A

    @staticmethod
    def backward(ctx, do, A):
        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        BLOCK = 64
        q, k, v, o, L, hash_ids0, hash_ids1, hash_ids2, keep_ids0, keep_ids1, keep_ids2, anchor_pos = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k.shape[2]
        do = do.contiguous()
        if sequence_parallel:
            replicas = triton.cdiv(seq_len_kv, BLOCK)
            new_dq_shape = (replicas, ) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros_like(q, dtype=q.dtype)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(triton.cdiv(q.shape[2], BLOCK) * ctx.grid[1], )](
            o,
            do,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1], triton.cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
            q, k, v, hash_ids0, hash_ids1, hash_ids2,
            keep_ids0, keep_ids1, keep_ids2, anchor_pos, ctx.sm_scale,  #
            o, do,  #
            dq, dk, dv,  #
            L,  #
            delta,  #
            o.numel(), q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            hash_ids0.stride(0),
            q.shape[0], q.shape[1], q.shape[2],  #
            q.shape[0] * q.shape[1] * q.shape[2],  #
            triton.cdiv(seq_len_kv, BLOCK) * q.shape[0] * q.shape[1] * q.shape[2],  #
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            SEQUENCE_PARALLEL=sequence_parallel,  #
            CAUSAL=ctx.causal,  #
            MMA_V3=MMA_V3,  #
            num_warps=8,  #
            num_stages=1  #
        )

        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return dq, dk, dv, None, None, None, None, None


attention = _attention.apply


def attention_fn(q, k, v, keep_ids, hash_ids, causal=False, output_attentions=False):
    BATCH, H, N_CTX, D_HEAD = q.shape
    sm_scale = 1.0 / math.sqrt(D_HEAD)
    y, A = attention(q, k, v, hash_ids, keep_ids, sm_scale, causal, output_attentions)
    # y = triton.ops.attention(q, k, v, False, sm_scale, True).permute(0, 2, 1, 3).contiguous()
    return y.permute(0, 2, 1, 3).contiguous(), A