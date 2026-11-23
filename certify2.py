"""Integer-exact certificate for large L (Shor and/or triangle multipliers).

A = Az/8 exactly (entries are multiples of 1/8).  d, mu are rounded to multiples of 2^-K.
M = A + diag(d) - sum mu_t T_t  is then an exact integer matrix times 2^-(K+1).
PSD proof:  2^(2RB) (M - tau I) - Rz^T Rz = N  (exact integers), and
            max_i sum_j |N_ij|  <=  2^(2RB) tau   =>   M - tau I + (stuff) ... => M >= 0.
Precisely: M = tau I + 2^-2RB (Rz^T Rz + N),  and  tau I + 2^-2RB N >= (tau - 2^-2RB ||N||_inf) I >= 0.
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
import flint
from fastbuild import build_fast, exact_entry

SIGNS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
K = 40          # multipliers are integers times 2^-K
RB = 40         # Cholesky factor scaled by 2^RB


def certify_int(L, d, cuts=(), mu=(), tau_rel=1e-10, built=None, verbose=True):
    t0 = time.time()
    Af, cut, E, den, r1, r2 = built if built is not None else build_fast(L)
    assert 8 % den == 0 or den == 8 or den in (1, 2, 4, 8), den
    # integer matrix S = 2^(K+1) * M
    mu_i = [int(np.floor(max(0.0, m) * 2**K)) for m in mu]
    # float M to find shift
    Mf = Af.copy()
    for (i, j, k, s), m in zip(cuts, mu_i):
        if m == 0:
            continue
        s1, s2, s3 = SIGNS[s]
        h = m / 2**(K + 1)
        Mf[i, j] -= s1 * h; Mf[j, i] -= s1 * h
        Mf[j, k] -= s2 * h; Mf[k, j] -= s2 * h
        Mf[i, k] -= s3 * h; Mf[k, i] -= s3 * h
    d = np.maximum(np.asarray(d, dtype=float), 0.0)
    lmin = np.linalg.eigvalsh(Mf + np.diag(d))[0]
    scale = float(np.abs(Mf).max() + d.max())
    tau = tau_rel * scale
    shift = max(0.0, -lmin) + 4 * tau
    d_i = [int(np.ceil((v + shift) * 2**K)) for v in d]
    # exact integer S = 2^(K+1) M
    Az8 = np.rint(Af * 8).astype(np.int64)          # exact since entries are multiples of 1/8
    # sanity: verify float->int exactness on cut cells
    for (i, j), v in cut.items():
        assert Fr(int(Az8[i, j]), 8) == v, (i, j, v)
    S = Az8.astype(object) * (2**(K + 1) // 8)
    for i in range(L):
        S[i, i] += 2 * d_i[i]
    for (i, j, k, s), m in zip(cuts, mu_i):
        if m == 0:
            continue
        s1, s2, s3 = SIGNS[s]
        S[i, j] -= s1 * m; S[j, i] -= s1 * m
        S[j, k] -= s2 * m; S[k, j] -= s2 * m
        S[i, k] -= s3 * m; S[k, i] -= s3 * m
    # float Cholesky of M - tau I
    Mfl = S.astype(float) / 2**(K + 1)
    Rf = np.linalg.cholesky(Mfl - tau * np.eye(L)).T
    Rz = flint.fmpz_mat([[int(round(v)) for v in row] for row in (Rf * 2**RB)])
    RtR = Rz.transpose() * Rz
    # tau as integer at scale 2^(K+1)
    tau_i = int(np.floor(tau * 2**(K + 1)))
    # N = 2^(2RB - K - 1) * (S - tau_i I) - RtR   (requires 2RB >= K+1)
    sh = 2 * RB - (K + 1)
    assert sh >= 0
    Sz = flint.fmpz_mat([[int(S[i, j]) - (tau_i if i == j else 0) for j in range(L)] for i in range(L)])
    N = Sz * (flint.fmpz(2) ** sh) - RtR
    ent = N.entries()
    maxrow = 0
    for i in range(L):
        r = sum(abs(int(x)) for x in ent[i * L:(i + 1) * L])
        maxrow = max(maxrow, r)
    bound = tau_i * 2**sh                 # 2^(2RB) * tau_q where tau_q = tau_i / 2^(K+1)
    ok = maxrow <= bound
    sum_d = Fr(sum(d_i), 2**K); sum_mu = Fr(sum(mu_i), 2**K)
    lam0 = -(sum_d + sum_mu) / (L * L) - E
    beta0 = Fr(1, 16) + lam0 / 4
    if verbose:
        print(f"L={L}: exact PSD check {'PASSED' if ok else 'FAILED'} (||N||/bound = {maxrow/bound:.2e}); shift {shift:.2e}; "
              f"lambda0 = {float(lam0):.8f}; delta_3 >= {float(beta0):.8f}   [{time.time()-t0:.0f}s]", flush=True)
    return ok, lam0, beta0, d_i, mu_i


if __name__ == "__main__":
    from shor import mixing
    L = int(sys.argv[1])
    t = time.time()
    built = build_fast(L)
    Af = built[0]
    V, d, obj = mixing(Af, r=24, sweeps=600, tol=1e-12)
    np.save(f"shor_d_{L}.npy", d)
    print(f"L={L}: mixing done, primal {obj/L**2:.8f} [{time.time()-t:.0f}s]", flush=True)
    ok, lam0, beta0, d_i, mu_i = certify_int(L, d, built=built)
    with open(f"cert_shor_{L}.txt", "w") as f:
        f.write(f"L={L}\nok={ok}\nlambda0={lam0}\nbeta0={beta0}\nK={K}\nd_int={d_i}\n")
