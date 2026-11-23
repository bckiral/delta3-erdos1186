"""Cutting planes for Shor + triangle inequalities, keeping cuts (no cycling)."""
import sys, time
import numpy as np
import cvxpy as cp
from kernel import build

SIGNS = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])


def violated(X, tol=1e-6, maxn=40000, existing=None):
    L = X.shape[0]
    vals, ids = [], []
    for i in range(L - 2):
        for j in range(i + 1, L - 1):
            k = np.arange(j + 1, L)
            a, b, c = X[i, j], X[j, k], X[i, k]
            for si, s in enumerate(SIGNS):
                v = s[0] * a + s[1] * b + s[2] * c
                bad = np.nonzero(v < -1 - tol)[0]
                if len(bad):
                    vals.append(-1 - v[bad])
                    ids.append(np.stack([np.full(len(bad), i), np.full(len(bad), j), k[bad], np.full(len(bad), si)], 1))
    if not vals:
        return np.zeros((0, 4), dtype=np.int64)
    vals = np.concatenate(vals); ids = np.concatenate(ids)
    order = np.argsort(-vals)
    ids = ids[order]
    if existing is not None and len(existing):
        key = lambda r: ((r[:, 0] * L + r[:, 1]) * L + r[:, 2]) * 4 + r[:, 3]
        ex = set(key(existing).tolist())
        m = np.array([kk not in ex for kk in key(ids).tolist()])
        ids = ids[m]
    return ids[:maxn]


def solve(A, cuts, eps=1e-7):
    L = A.shape[0]
    X = cp.Variable((L, L), symmetric=True)
    cons = [X >> 0, cp.diag(X) <= 1]
    if len(cuts):
        I, J, K, S = cuts[:, 0], cuts[:, 1], cuts[:, 2], SIGNS[cuts[:, 3]]
        tri = cp.multiply(S[:, 0], X[I, J]) + cp.multiply(S[:, 1], X[J, K]) + cp.multiply(S[:, 2], X[I, K])
        cons.append(tri >= -1)
    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), cons)
    prob.solve(solver="SCS", eps=eps, max_iters=300000)
    mu = cons[2].dual_value if len(cuts) else np.zeros(0)
    return prob.value, X.value, cons[1].dual_value, mu


if __name__ == "__main__":
    L = int(sys.argv[1]); rounds = int(sys.argv[2]); addn = int(sys.argv[3])
    A, E = build(L)
    t = time.time()
    val, X, d, mu = solve(A, np.zeros((0, 4), dtype=np.int64))
    cuts = np.zeros((0, 4), dtype=np.int64)
    idle = np.zeros(0, dtype=np.int64)
    for r in range(rounds):
        new = violated(X, maxn=addn, existing=cuts)
        if len(new) == 0:
            print("  no violated cuts: converged", flush=True)
            break
        if len(cuts):
            idle = np.where(mu > 1e-8, 0, idle + 1)
            keep = idle < 3
            cuts, idle = cuts[keep], idle[keep]
        cuts = np.concatenate([cuts, new]); idle = np.concatenate([idle, np.zeros(len(new), dtype=np.int64)])
        val, X, d, mu = solve(A, cuts)
        w = np.linalg.eigvalsh(X)
        print(f"  round {r}: cuts {len(cuts)} active {(mu > 1e-8).sum()}  value {val/L**2:.6f}  "
              f"rank-ish {w[-1]:.2f}/{w[-2]:.3f}  [{time.time()-t:.0f}s]", flush=True)
    np.save(f"tri2_X_{L}.npy", X); np.save(f"tri2_cuts_{L}.npy", cuts); np.save(f"tri2_mu_{L}.npy", mu); np.save(f"tri2_d_{L}.npy", d)
    print(f"L={L}: final value {val/L**2:.6f}", flush=True)
