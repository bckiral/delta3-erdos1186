"""Shor (MAX-CUT SDP) lower bound for lambda* = min_g ∬ H g g.

min_{x in [-1,1]^L} x^T A x >= -sum(d)  whenever A + diag(d) is PSD.
Solve the primal SDP by the mixing method (Burer–Monteiro coordinate descent),
read off d from stationarity, repair PSD by a uniform shift.
"""
import sys, time
import numpy as np
from kernel import build


def mixing(A, r=24, sweeps=400, tol=1e-10, seed=0, verbose=False):
    L = A.shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((L, r))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Ad = np.diag(A).copy()
    G = A @ V
    prev = np.inf
    for s in range(sweeps):
        for i in range(L):
            g = G[i] - Ad[i] * V[i]
            nv = -g / np.linalg.norm(g)
            dv = nv - V[i]
            if np.any(dv):
                G += np.outer(A[:, i], dv)
                V[i] = nv
        obj = float(np.sum(G * V))
        if verbose and s % 50 == 0:
            print(f"   sweep {s}: obj {obj:.10f}", flush=True)
        if abs(prev - obj) < tol * max(1, abs(obj)):
            break
        prev = obj
    G = A @ V
    gnorm = np.linalg.norm(G - Ad[:, None] * V, axis=1)
    d = gnorm - Ad
    return V, d, obj


def shor_bound(L, r=24, verbose=False):
    t = time.time()
    A, E = build(L)
    V, d, obj = mixing(A, r=r, verbose=verbose)
    lmin = np.linalg.eigvalsh(A + np.diag(d))[0]
    shift = max(0.0, -lmin) * 1.0000001 + 1e-12
    dsum = d.sum() + L * shift
    lam_lb = -dsum / L**2 - E
    lam_sdp = obj / L**2
    print(f"L={L}: SDP primal {lam_sdp:.6f}, dual {-dsum/L**2:.6f} (shift {shift:.2e}), E_L={E:.6f} "
          f"=> lambda* >= {lam_lb:.6f}, delta_3 >= {1/16 + lam_lb/4:.6f}   [{time.time()-t:.0f}s]", flush=True)
    return A, E, V, d


if __name__ == "__main__":
    for L in [int(a) for a in sys.argv[1:]]:
        shor_bound(L)
