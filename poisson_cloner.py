from typing import Literal, Optional, Tuple
import time
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from utils import _read_color_float64, _read_mask_bool, BuildInfo

Mode = Literal["seamless", "mixed"]

class PoissonCloner:
    def __init__(self, src_path: str, target_path: str, mask_path: str):
        self.src = _read_color_float64(src_path)
        self.target = _read_color_float64(target_path)
        self.mask = _read_mask_bool(mask_path)

        if self.src.shape[:2] != self.target.shape[:2]:
            raise ValueError("Source and target must have the same spatial shape.")
        if self.mask.shape != self.target.shape[:2]:
            raise ValueError("Mask must have the same spatial shape as images.")

        self.h, self.w = self.mask.shape
        
        # 【给分点B.1：mask 内像素编号】
        # N为mask内白色像素的数量，即待求解的未知量个数
        # 因此A的规模为 N×N
        # im2var[y, x]把mask内的映射为[0, N-1]；mask外为-1
        self.im2var = np.full((self.h, self.w), -1, dtype=np.int32)
        ys, xs = np.nonzero(self.mask)
        self.im2var[ys, xs] = np.arange(ys.size, dtype=np.int32)
        self.mask_ys = ys.astype(np.int32)
        self.mask_xs = xs.astype(np.int32)
        self.n = int(ys.size)

        self.a: Optional[sp.csr_matrix] = None
        self.build_info: Optional[BuildInfo] = None

    def build_matrix_A(self):
        if self.a is not None and self.build_info is not None:
            return self.build_info

        t0 = time.perf_counter()
        # 【给分点B.2：稀疏矩阵A构造】
        # A是稀疏矩阵，采用三元组表示，A[rows[k], cols[k]]=data[k]
        rows = []
        cols = []
        data = []

        # 离散拉普拉斯：对 mask 内每个像素 p，左端是 degree(p)*f_p - sum_{q in N_p ∩ W} f_q
        for y, x in zip(self.mask_ys, self.mask_xs):
            i = int(self.im2var[y, x])
            degree = 0

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yn = y + dy
                xn = x + dx
                # 某个方向的坐标超出了图片的物理边界，那么不需要加上这一项，同时$f(y,x)$前的系数也要相应修改
                if yn < 0 or yn >= self.h or xn < 0 or xn >= self.w:
                    continue
                degree += 1
                j = int(self.im2var[yn, xn])
                if j >= 0:
                    # 某个方向的坐标在mask内才设置为-1，否则放到等式右边的b
                    rows.append(i)
                    cols.append(j)
                    data.append(-1.0)

            rows.append(i)
            cols.append(i)
            data.append(float(degree))
        # 接受三元组系数矩阵，构建出完整的稀疏矩阵，并转换为适合高速求解的CSR格式
        a = sp.coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(self.n, self.n),
        ).tocsr()
        t1 = time.perf_counter()

        self.a = a
        self.build_info = BuildInfo(
            n_unknowns=self.n,
            nnz=int(a.nnz),
            build_time_s=float(t1 - t0),
        )
        return self.build_info

    def _build_b_channel(self, ch, mode) -> np.ndarray:
        '''
        逐通道构建右端向量 b，方程为 A * f = b；
        输入：通道ch，mode="seamless"或"mixed"
        输出：右端向量b，长度为N
        '''
        b = np.zeros(self.n, dtype=np.float64)

        g = self.src[..., ch]
        f_star = self.target[..., ch]

        for y, x in zip(self.mask_ys, self.mask_xs):
            i = int(self.im2var[y, x])
            rhs = 0.0
            gp = float(g[y, x])
            tp = float(f_star[y, x])

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yn = y + dy
                xn = x + dx
                if yn < 0 or yn >= self.h or xn < 0 or xn >= self.w:
                    continue
                gq = float(g[yn, xn])
                tq = float(f_star[yn, xn])

                if mode == "seamless":
                    # importing gradients，v_{pq}=g_p-g_q
                    v_pq = gp - gq
                else:
                    # mixed gradient
                    # 【给分点C.1：逐通道mixed gradient选择规则】 
                    # 在每条有向边上对比 |target 梯度|与|source 梯度|，选更大的那个
                    grad_t = tp - tq
                    grad_s = gp - gq
                    if abs(grad_t) > abs(grad_s):
                        v_pq = grad_t
                    else:
                        v_pq = grad_s
                rhs += v_pq
                # 若q不在mask内，那它就在\partial\Omega，则f_q固定为f*_q，作为Dirichlet边界项移到右端
                if self.im2var[yn, xn] < 0:
                    rhs += tq

            b[i] = rhs
        return b

    def solve(self, mode: Mode) -> Tuple[np.ndarray, BuildInfo, float]:
        info = self.build_matrix_A()
        if self.a is None:
            raise RuntimeError("Matrix A is not built.")

        t0 = time.perf_counter()
        out = self.target.copy()
        # 【给分点B.3：RGB三通道求解】
        # 【给分点C.2：实现正确】
        # 不同通道，A共用，b按通道分别构建并独立求解
        for ch in range(3):
            b = self._build_b_channel(ch, mode=mode)
            x = spsolve(self.a, b).astype(np.float64)
            out[..., ch][self.mask] = x

        out = np.clip(out, 0.0, 1.0)
        t1 = time.perf_counter()
        return out, info, float(t1 - t0)