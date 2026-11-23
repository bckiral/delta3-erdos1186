"""Lagrangian hybrid bound: fine-grid Shor + coarse-grid triangle multipliers.

For nu >= 0 on coarse triangles t (coarse grid Lc, fine grid Lf = m*Lc):
   B(nu) = A_f / Lf^2 - P^T C(nu) P,   C(nu) = sum_t nu_t T_t,   (P = block averaging)
   lambda* >= -(sum d)/1 - sum nu - E_Lf     whenever B(nu) + diag(d) >= 0, d >= 0.
(Here A_f/Lf^2 is used so all quantities are O(1); d is then the Shor dual of B.)
Inner problem: mixing method on B(nu) with block-structured updates.
Outer: projected subgradient on nu (subgradient = triangle violation of the coarse aggregate).
"""
import sys, time
import numpy as np
from fastbuild import build_fast

SIGNS = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=float)


def C_of(nu, cuts, Lc):
    C = np.zeros((Lc, Lc))
    I, J, K, S = cuts[:, 0], cuts[:, 1], cuts[:, 2], SIGNS[cuts[:, 3]]
    np.add.at(C, (I, J), nu * S[:, 0] / 2); np.add.at(C, (J, I), nu * S[:, 0] / 2)
    np.add.at(C, (J, K), nu * S[:, 1] / 2); np.add.at(C, (K, J), nu * S[:, 1] / 2)
    np.add.at(C, (I, K), nu * S[:, 2] / 2); np.add.at(C, (K, I), nu * S[:, 2] / 2)
    return C


class Inner:
    def __init__(self, Af_n, Lc, r=24, seed=0):
        self.A = Af_n; self.Lf = Af_n.shape[0]; self.Lc = Lc; self.m = self.Lf // Lc
        rng = np.random.default_rng(seed)
        V = rng.standard_normal((self.Lf, r)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        self.V = V
        self.blk = np.arange(self.Lf) // self.m

    def B_diag(self, C):
        return np.diag(self.A) - C[self.blk, self.blk] / self.m**2

    def solve(self, C, sweeps=30, tol=1e-9):
        A, V, m, blk = self.A, self.V, self.m, self.blk
        Bd = self.B_diag(C)
        GA = A @ V
        W = np.zeros((self.Lc, V.shape[1])); np.add.at(W, blk, V)
        Q = C @ W
        prev = None
        for s in range(sweeps):
            for i in range(self.Lf):
                b = blk[i]
                g = GA[i] - Q[b] / m**2 - Bd[i] * V[i]
                nv = -g / np.linalg.norm(g)
                dv = nv - V[i]
                GA += np.outer(A[:, i], dv)
                W[b] += dv
                Q += np.outer(C[:, b], dv)
                V[i] = nv
            G = GA - Q[blk] / m**2
            obj = float(np.sum(G * V))
            if prev is not None and abs(prev - obj) < tol * abs(obj):
                break
            prev = obj
        G = GA - Q[blk] / m**2
        gnorm = np.linalg.norm(G - Bd[:, None] * V, axis=1)
        d = gnorm - Bd
        self.V = V
        Wn = W / m
        return obj, d, Wn @ Wn.T

    def certified(self, C, d):
        Bm = self.A - np.kron(C, np.ones((self.m, self.m))) / self.m**2
        lmin = np.linalg.eigvalsh(Bm + np.diag(d))[0]
        return d.sum() + self.Lf * max(0.0, -lmin), lmin


def viol(Xc, cuts):
    I, J, K, S = cuts[:, 0], cuts[:, 1], cuts[:, 2], SIGNS[cuts[:, 3]]
    return -1 - (S[:, 0] * Xc[I, J] + S[:, 1] * Xc[J, K] + S[:, 2] * Xc[I, K])


def all_triangles(Lc):
    i, j, k = np.array([(i, j, k) for i in range(Lc) for j in range(i + 1, Lc) for k in range(j + 1, Lc)]).T
    n = len(i)
    return np.stack([np.repeat(i, 4), np.repeat(j, 4), np.repeat(k, 4), np.tile(np.arange(4), n)], 1)


if __name__ == "__main__":
    Lf, Lc, iters = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    init = sys.argv[4] if len(sys.argv) > 4 else None
    Af, cut, E, den, r1, r2 = build_fast(Lf)
    E = float(E)
    An = Af / Lf**2
    cuts = all_triangles(Lc)
    nu = np.zeros(len(cuts))
    if init:  # initialise from a coarse triangle dual (tri2 output at L = Lc)
        c0 = np.load(f"tri2_cuts_{init}.npy"); mu0 = np.load(f"tri2_mu_{init}.npy")
        idx = {tuple(c): n for n, c in enumerate(cuts.tolist())}
        for c, mval in zip(c0.tolist(), mu0):
            if mval > 0:
                nu[idx[tuple(c)]] = mval / Lc**2
    inner = Inner(An, Lc)
    t = time.time()
    best = -np.inf
    step0 = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-5
    for it in range(iters):
        C = C_of(nu, cuts, Lc)
        obj, d, Xc = inner.solve(C, sweeps=int(sys.argv[6]) if (it == 0 and len(sys.argv) > 6) else (60 if it == 0 else 8))
        qv = obj - nu.sum()                     # dual function estimate (inner value - sum nu)
        if it % 10 == 0 or it == iters - 1:
            dsum, lmin = inner.certified(C, np.maximum(d, 0))
            lb = -dsum - nu.sum() - E
            best = max(best, lb)
            print(f"  it {it}: inner {obj:.6f} sum nu {nu.sum():.6f} q {qv:.6f} | certified lambda >= {lb:.6f} "
                  f"(delta3 >= {1/16 + lb/4:.6f}); best delta3 >= {1/16 + best/4:.6f}; active {(nu>0).sum()} [{time.time()-t:.0f}s]", flush=True)
            np.save(f"lag_nu_{Lf}_{Lc}.npy", nu)
        v = viol(Xc, cuts)
        step = step0 / np.sqrt(1 + it / 20)
        nu = np.maximum(0.0, nu + step * v)
