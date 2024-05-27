import os
import pickle as pkl
import random

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops
import numpy as np
from tqdm import tqdm

# 任意两个基向量夹角cos绝对值的最大值
# The maximum value of the absolute cosine value of the angle between any two basis vectors
THRESHOLD = 0.002

# 类别数量，也可以理解为需要生成基向量的数量
# Number of categories, which can also be interpreted as the number of basis vectors N that need to be generated, num_cls >= N
NUM_CLS = 1000

# 基向量的维度
# Dimension for basis vectors
DIM = 1000

# 由于显存不够，所以需要切片优化。每次切片的大小。 slice_size<num_cls
# Slicing optimization is required due to insufficient memory
SLICE_SIZE = 130

# 优化的step数量，达到threshold会立即退出
# Optimize step numbers
STEP = 100000
LR = 1e-3  # learning rate
SAVE_NAME = 'eq_1000_1000.pkl'  # pkl_name: eq_dim_numcls


def main(num_cls, dim, slice_size, threshold):
    basis_vec = ms.Parameter(ops.L2Normalize(1)(ops.standard_normal((num_cls, dim))), name='basis_vec')
    optim = nn.SGD([basis_vec], learning_rate=LR)
    matmul = ops.MatMul(transpose_b=True)

    def forward_fn(a, b, e, thr):
        m = matmul(a, b).abs() - e
        loss = ops.relu(m - thr).sum()
        return loss, m

    grad_fn = ops.value_and_grad(forward_fn, 1, [basis_vec], has_aux=True)
    pbar = tqdm(range(STEP), total=STEP)
    for _ in pbar:
        basis_vec.set_data(ops.L2Normalize(1)(basis_vec.data))
        mx = threshold
        grads = msnp.zeros_like(basis_vec)
        for i in range((num_cls - 1) // slice_size + 1):
            start = slice_size * i
            end = min(slice_size * (i + 1), num_cls)
            e = ops.one_hot(msnp.arange(start, end), num_cls)
            (loss, m), grads_partial = grad_fn(basis_vec[start:end], basis_vec, e, threshold)
            mx = max(mx, m.max().asnumpy().tolist())
            grads = grads + grads_partial[0]

        if mx <= threshold + 0.0001:
            pkl.dump(basis_vec.data, open(SAVE_NAME, 'bw'))
            return
        optim((grads,))
        pbar.set_description(f'{mx:.4f}')


if __name__ == '__main__':
    if not os.path.exists(SAVE_NAME):
        seed = 42
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        ms.common.seed.set_seed(seed)
        context.set_context(device_target="GPU")
        main(NUM_CLS, DIM, SLICE_SIZE, THRESHOLD)
    else:
        basis_vec = pkl.load(open(SAVE_NAME, 'rb'))
        print(basis_vec.shape)
        context.set_context(device_target="GPU")
        m_max = 0.
        m_min = 1.
        ada = 0.
        for j in tqdm(range(NUM_CLS)):
            m = (basis_vec[j] @ basis_vec.T).abs() - ops.one_hot(ms.tensor(j), NUM_CLS)
            ada += m.sum().item() / (NUM_CLS - 1)
            if m.max() > m_max:
                m_max = m.max()
            if m.min() < m_min:
                m_min = m.min()
        ada /= NUM_CLS
        print(ada)
        print(m_max, m_min)
