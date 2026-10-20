import torch

import triton
import triton.language as tl

"""
STAGE==1:
计算下三角矩阵的所有块(不含对角线)

STAGE==2:
计算对角线上的所有块

STAGE==3:
不做causal mask计算

把连续的一个计算阶段拆成了2个STAGE
如果在计算途中进行if条件判断会极大降低并行效率,我们解决的办法就是把条件判断提到计算之前,也就是这种多STAGE来解决
"""

@triton.jit                                                         # 这个函数是kernel内部的辅助函数，必须用@triton.jit装饰器修饰。和kernel共享线程块
def _attn_fwd_inner(
    O_block,
    m_i,                                                            # 这个m_i是这一个QK_block之前的同一行QK_block中的最大值(还不是这一行QK_block中的global_maximum)
    l_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale:tl.constexpr,
    offsets_q:tl.constexpr,
    offsets_kv:tl.constexpr,
    BLOCK_SIZE_Q:tl.constexpr,
    BLOCK_SIZE_KV:tl.constexpr,
    STAGE:tl.constexpr,
    SEQ_LEN:tl.constexpr
):
    if STAGE==1:
        lo,hi=0,block_index_q*BLOCK_SIZE_Q
    elif STAGE==2:
        lo,hi=block_index_q*BLOCK_SIZE_Q,(block_index_q+1)*BLOCK_SIZE_Q
    
    else:
        lo,hi=0,SEQ_LEN

    K_block_ptr=tl.advance(K_block_ptr,(0,lo))
    V_block_ptr=tl.advance(V_block_ptr,(lo,0))

    for start_kv in range(lo,hi,BLOCK_SIZE_KV):

        start_kv=tl.multiple_of(start_kv,BLOCK_SIZE_KV)                              # 这一步在表述start_kv是BLOCK_SIZE_KV的倍数，方便compiler优化                                      
        
        K_block=tl.load(K_block_ptr)
        QK_block=tl.dot(Q_block,K_block)

        if STAGE==2:
            mask=offsets_q[:,None]>=(start_kv+offsets_kv)[None,:]                    # 这一步用broadcasting
            QK_block=QK_block*softmax_scale+tl.where(mask,0,-1.0e6)           # 类似if-else语句(和torch中一致)，True执行0,False执行float("-inf")/-1.0e6
            m_ij=tl.maximum(m_i,tl.max(QK_block,axis=1))
            QK_block-=m_ij[:,None]

        else:
            QK_block=QK_block*softmax_scale
            m_ij=tl.maximum(m_i,tl.max(QK_block,axis=1))
            QK_block-=m_ij[:,None]

        alpha=tl.exp(m_i-m_ij)

        P_block=tl.exp(QK_block)

        l_ij=tl.sum(P_block,axis=1)

        l_i=l_i*alpha+l_ij

        V_block=tl.load(V_block_ptr)

        P_block=P_block.to(tl.float16)

        PV_block=tl.dot(P_block,V_block)

        O_block=O_block*alpha[:,None]+PV_block

        m_i=m_ij

        K_block_ptr=tl.advance(K_block_ptr,(0,BLOCK_SIZE_KV))
        V_block_ptr=tl.advance(V_block_ptr,(BLOCK_SIZE_KV,0))


    return O_block,l_i,m_i


        
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        softmax_scale,
        O,                                              # O是最后输出的output，也需要在显存上预先申请一块空间用来保存数据
        M,
        stride_Q_batch,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_head,
        stride_K_seq,
        stride_K_dim,
        stride_V_batch,
        stride_V_head,
        stride_V_seq,
        stride_V_dim,
        stride_O_batch,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        BATCH_SIZE:tl.constexpr,
        NUM_HEADS:tl.constexpr,
        SEQ_LEN:tl.constexpr,
        HEAD_DIM:tl.constexpr,
        BLOCK_SIZE_Q:tl.constexpr,
        BLOCK_SIZE_KV:tl.constexpr,
        STAGE:tl.constexpr
):
    block_index_q=tl.program_id(axis=0)

    index_batch_head=tl.program_id(1)

    index_batch=index_batch_head//NUM_HEADS

    index_head=index_batch_head%NUM_HEADS

    qvk_offsets=index_batch.to(tl.int64)*stride_Q_batch+index_head.to(tl.int64)*stride_Q_head


    # 理解这一步的关键是不要把张量看成多维而是一维连续在显存上存储的数组
    # make_block_ptr()返回的不只是数据块的首地址，他会把所有数据的地址按照你的传参需求封装起来，可以理解成封装成了一个二维张量
    Q_block_ptr=tl.make_block_ptr(
        base=Q+qvk_offsets,                             # 表示当前正在处理的头的首地址
        shape=(SEQ_LEN,HEAD_DIM),                       # 和stride联合使用，目的是为了防止访问到不是当前头的元素
        strides=(stride_Q_seq,stride_Q_dim),            # --------------------------------------------------
        offsets=(block_index_q*BLOCK_SIZE_Q,0),         # 这一步是针对当前线程块来说的，表示当前线程块所需的Q矩阵块的首坐标
        block_shape=(BLOCK_SIZE_Q,HEAD_DIM),            # 表示当前线程块所需的数据块大小
        order=(1,0)                                     # 表示我们所处理的矩阵是row-major的
                                                        # 这一步指针指向的矩阵形状是(seq_len,head_dim),也就是说我们直接让它指向了整个头
    )

    V_block_ptr=tl.make_block_ptr(
        base=V+qvk_offsets,
        shape=(SEQ_LEN,HEAD_DIM),
        strides=(stride_V_seq,stride_V_dim),
        offsets=(0,0),
        block_shape=(BLOCK_SIZE_KV,HEAD_DIM),
        order=(1,0)
    )
    # 我们可以利用此函数直接“转置地”读取数据
    K_block_ptr=tl.make_block_ptr(
        base=K+qvk_offsets,                             
        shape=(HEAD_DIM,SEQ_LEN),                       # shape会改变逻辑存储，也就是说当你tl.load()之后只用处理一个(seq_len,head_dim)形状的矩阵而不是(B,H,seq_len,head_dim)
        strides=(stride_K_dim,stride_K_seq),
        offsets=(0,0),
        block_shape=(HEAD_DIM,BLOCK_SIZE_KV),
        order=(1,0)
    )

    O_block_ptr=tl.make_block_ptr(
        base=O+qvk_offsets,
        shape=(SEQ_LEN,HEAD_DIM),
        strides=(stride_O_seq,stride_O_dim),
        offsets=(block_index_q*BLOCK_SIZE_Q,0),
        block_shape=(BLOCK_SIZE_Q,HEAD_DIM),
        order=(1,0)
    )

    offsets_q=block_index_q*BLOCK_SIZE_Q+tl.arange(0,BLOCK_SIZE_Q)          # 表示我当前线程块需要取的token的id(在Query中)

    offsets_kv=tl.arange(0,BLOCK_SIZE_KV)                                   # 表示for循环中每次要取的token的偏移量(在Key,Value中),需要搭配start_kv使用

    m_i=tl.zeros([BLOCK_SIZE_Q],dtype=tl.float32) - float("inf")

    l_i=tl.zeros([BLOCK_SIZE_Q],dtype=tl.float32) + 1.0

    O_block=tl.zeros([BLOCK_SIZE_Q,HEAD_DIM],dtype=tl.float32)              # 这个block用于存放和累加数据，相当于accumulator

    Q_block=tl.load(Q_block_ptr)

    if STAGE==1 or STAGE==3:

        O_block,l_i,m_i=_attn_fwd_inner(O_block,
                        m_i,
                        l_i,
                        Q_block,
                        K_block_ptr,V_block_ptr,
                        block_index_q,
                        softmax_scale,
                        offsets_q,
                        offsets_kv,
                        BLOCK_SIZE_Q,
                        BLOCK_SIZE_KV,
                        4-STAGE,
                        SEQ_LEN
                        )

    if STAGE==3:

        O_block,l_i,m_i=_attn_fwd_inner(O_block,
                        m_i,
                        l_i,
                        Q_block,
                        K_block_ptr,V_block_ptr,
                        block_index_q,
                        softmax_scale,
                        offsets_q,
                        offsets_kv,
                        BLOCK_SIZE_Q,
                        BLOCK_SIZE_KV,
                        2,
                        SEQ_LEN
                        )

    lse=m_i+tl.math.log(l_i)

    m_ptr=M+index_batch_head*SEQ_LEN+offsets_q
    tl.store(m_ptr,lse)
    O_block=O_block/l_i[:,None]
    tl.store(O_block_ptr,O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    BATCH_SIZE,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_SIZE_Q:tl.constexpr,
    HEAD_DIM:tl.constexpr                                        # O和V是一个形状的,dO和O是一个形状的
):
    block_index_q=tl.program_id(0)
    index_batch_head=tl.program_id(1)

    offsets_q=tl.arange(0,BLOCK_SIZE_Q)+block_index_q*BLOCK_SIZE_Q

    O_block_ptr=tl.make_block_ptr(
        O+index_batch_head*SEQ_LEN*HEAD_DIM,
        shape=(SEQ_LEN,HEAD_DIM),
        strides=(HEAD_DIM,1),
        offsets=(block_index_q*BLOCK_SIZE_Q,0),
        block_shape=(BLOCK_SIZE_Q,HEAD_DIM),
        order=(1,0)
    )
    O_block=tl.load(O_block_ptr)

    dO_block_ptr=tl.make_block_ptr(
        dO+index_batch_head*SEQ_LEN*HEAD_DIM,
        shape=(SEQ_LEN,HEAD_DIM),
        strides=(HEAD_DIM,1),
        offsets=(block_index_q*BLOCK_SIZE_Q,0),
        block_shape=(BLOCK_SIZE_Q,HEAD_DIM),
        order=(1,0)
    )
    dO_block=tl.load(dO_block_ptr).to(tl.float32)

    D_block=tl.sum(dO_block*O_block,axis=1)
    D_block_ptr=D+index_batch_head*SEQ_LEN+offsets_q

    tl.store(D_block_ptr,D_block)

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    # We access the K and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # Autoregressive masking.
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Iterates over the sequence dimension of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking.
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )  # Shape: (BLOCK_KV1, BLOCK_Q1)
            # Replace all the masked values with 0.
            # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in "m")
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,causal:bool,softmax_scale):
        HEAD_DIM_Q=Q.shape[-1]
        HEAD_DIM_K=K.shape[-1]
        HEAD_DIM_V=V.shape[-1]

        BATCH_SIZE,NUM_HEADS,SEQ_LEN=Q.shape[:-1]

        assert HEAD_DIM_K==HEAD_DIM_Q and HEAD_DIM_Q==HEAD_DIM_V

        BLOCK_SIZE_Q=16
        BLOCK_SIZE_KV=16

        O=torch.empty_like(Q)
        STAGE=3 if causal else 1

        grid=lambda meta:(triton.cdiv(SEQ_LEN,meta['BLOCK_SIZE_Q']),BATCH_SIZE*NUM_HEADS)

        M=torch.empty([BATCH_SIZE,NUM_HEADS,SEQ_LEN],dtype=torch.float32,device=Q.device)

        _attn_fwd[grid](
                Q,
                K,
                V,
                softmax_scale,
                O,
                M,
                Q.stride(0),
                Q.stride(1),
                Q.stride(2),
                Q.stride(3),
                K.stride(0),
                K.stride(1),
                K.stride(2),
                K.stride(3),
                V.stride(0),
                V.stride(1),
                V.stride(2),
                V.stride(3),
                O.stride(0),
                O.stride(1),
                O.stride(2),
                O.stride(3),
                BATCH_SIZE,
                NUM_HEADS,
                SEQ_LEN,
                HEAD_DIM_Q,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                STAGE
        )
        ctx.save_for_backward(Q,K,V,O,M)
        ctx.grid=grid
        ctx.softmax_scale=softmax_scale
        ctx.HEAD_DIM=HEAD_DIM_K
        ctx.causal=causal

        return O

    @staticmethod
    def backward(ctx, dO):                                      # dO就是上游传下来的dL/dO(当然这里应该写作partial)

        Q,K,V,O,M=ctx.saved_tensors                             # saved_tensors是ctx的一个成员,保存ctx.saved_for_backward()传入的参数

        dQ=torch.empty_like(Q)
        dK=torch.empty_like(K)
        dV=torch.empty_like(V)

        D=torch.empty_like(M)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        
        preprocess_grid=(SEQ_LEN//BLOCK_SIZE_MACRO,BATCH_SIZE*NUM_HEADS)

        _attn_bwd_preprocess[preprocess_grid](
            O,
            dO,
            D,
            BATCH_SIZE,
            NUM_HEADS,
            SEQ_LEN,
            BLOCK_SIZE_MACRO,
            ctx.HEAD_DIM
        )
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # Fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_op(BATCH_SIZE=2, NUM_HEADS=4, SEQ_LEN=512, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=2, NUM_HEADS=4, SEQ_LEN=512, HEAD_DIM=64, causal=False)
    print("PASSED")
       