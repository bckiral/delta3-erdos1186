"""Exact certificate for the hybrid bound (fine Shor + coarse triangle multipliers).

Claim:  for every g:[0,1]->[-1,1] (in particular every +-1 colouring limit),
    J(g) >= x^T (A_f/Lf^2) x - E_Lf,   x = fine block means in [-1,1]^Lf,
and for x in [-1,1]^Lf, with xc = P x in [-1,1]^Lc:
    x^T (A_f/Lf^2) x = x^T M x - sum_i d_i x_i^2 + sum_t nu_t T_t(xc)  >=  - sum d - sum nu,
provided  M := A_f/Lf^2 + diag(d) - P^T C(nu) P  is PSD,  d, nu >= 0,
and T_t(xc) = s1 xc_i xc_j + s2 xc_j xc_k + s3 xc_i xc_k >= -1 on the cube (s1 s2 s3 = 1).
We verify  M' = Lf^2 M = A_f + diag(Lf^2 d) - Lc^2 kron(C, J_m)  PSD exactly.
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
import flint
from fastbuild import build_fast
from lag import Inner, C_of, all_triangles, SIGNS

K = 44
RB = 40


def main(Lf, Lc, nu_src, sweeps=600, tau_rel=1e-10):
    t0 = time.time()
    Af, cut, E, den, r1, r2 = build_fast(Lf)
    assert den in (1, 2, 4, 8)
    m = Lf // Lc
    cuts = all_triangles(Lc)
    if nu_src == "tri":
        c0 = np.load(f"tri2_cuts_{Lc}.npy"); mu0 = np.load(f"tri2_mu_{Lc}.npy")
        idx = {tuple(c): n for n, c in enumerate(cuts.tolist())}
        nu = np.zeros(len(cuts))
        for c, mv in zip(c0.tolist(), mu0):
            if mv > 0:
                nu[idx[tuple(c)]] = mv / Lc**2
    else:
        nu = np.load(nu_src)
    # rationalise nu (round down, nonnegative)
    nu_i = np.floor(np.maximum(nu, 0) * 2**K).astype(np.int64)
    nu_r = nu_i / 2**K
    keep = nu_i > 0
    cuts_k, nu_ik = cuts[keep], nu_i[keep]
    # exact coarse C as integers at scale 2^(K+1):  C2 = 2^(K+1) C
    C2 = np.zeros((Lc, Lc), dtype=np.int64)
    I, J, Kk, S = cuts_k[:, 0], cuts_k[:, 1], cuts_k[:, 2], SIGNS[cuts_k[:, 3]].astype(np.int64)
    for (a, b, col) in ((I, J, 0), (J, Kk, 1), (I, Kk, 2)):
        np.add.at(C2, (a, b), nu_ik * S[:, col]); np.add.at(C2, (b, a), nu_ik * S[:, col])
    C = C2 / 2**(K + 1)
    # inner Shor solve for d
    An = Af / Lf**2
    inner = Inner(An, Lc)
    obj, d, Xc = inner.solve(C, sweeps=sweeps, tol=1e-12)
    d = np.maximum(d, 0)
    Bm = An - np.kron(C, np.ones((m, m))) / m**2
    lmin = np.linalg.eigvalsh(Bm + np.diag(d))[0]
    scale = float(np.abs(Bm).max() + d.max())
    tau = tau_rel * scale
    shift = max(0.0, -lmin) + 4 * tau
    # d' = Lf^2 d rounded up to multiples of 2^-K
    dp_i = np.ceil((d + shift) * Lf**2 * 2**K).astype(np.int64)
    # S = 2^(K+1) M'  (integers):  A_f*2^(K+1) + 2*dp_i on diagonal - Lc^2 * kron(C2, J)
    A8 = np.rint(Af * 8).astype(np.int64)
    Sm = A8 * (2**(K + 1) // 8) - (Lc**2) * np.kron(C2, np.ones((m, m), dtype=np.int64))
    Sm[np.diag_indices(Lf)] += 2 * dp_i
    assert np.abs(Sm).max() < 2**62
    tau_p = tau * Lf**2                         # tau in M' units
    Mfl = Sm.astype(float) / 2**(K + 1)
    Rf = np.linalg.cholesky(Mfl - tau_p * np.eye(Lf)).T
    print(f"  float Cholesky ok, building exact residual... [{time.time()-t0:.0f}s]", flush=True)
    Rz = flint.fmpz_mat([[int(v) for v in row] for row in np.rint(Rf * 2**RB).astype(np.int64)])
    RtR = Rz.transpose() * Rz
    tau_i = int(np.floor(tau_p * 2**(K + 1)))
    sh = 2 * RB - (K + 1)
    assert sh >= 0
    Sz = flint.fmpz_mat([[int(v) for v in row] for row in (Sm - tau_i * np.eye(Lf, dtype=np.int64))])
    N = Sz * (flint.fmpz(2) ** sh) - RtR
    ent = N.entries()
    maxrow = max(sum(abs(int(x)) for x in ent[i * Lf:(i + 1) * Lf]) for i in range(Lf))
    bound = tau_i * 2**sh
    ok = maxrow <= bound
    sum_dp = Fr(int(dp_i.sum()), 2**K)
    sum_nu = Fr(int(nu_ik.sum()), 2**K)
    lam0 = -sum_dp / (Lf * Lf) - sum_nu - E
    beta0 = Fr(1, 16) + lam0 / 4
    print(f"Lf={Lf} Lc={Lc}: exact PSD check {'PASSED' if ok else 'FAILED'} (||N||_inf/bound = {maxrow/bound:.2e}); "
          f"lambda0 = {float(lam0):.8f};  delta_3 >= {float(beta0):.8f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(f"cert_hybrid_{Lf}_{Lc}.npz", cuts=cuts_k, nu_int=nu_ik, dp_int=dp_i, K=K, Lf=Lf, Lc=Lc)
    with open(f"cert_hybrid_{Lf}_{Lc}.txt", "w") as f:
        f.write(f"Lf={Lf} Lc={Lc} ok={ok}\nlambda0={lam0}\nbeta0={beta0}\nE={E}\nsum_dp={sum_dp}\nsum_nu={sum_nu}\n")
    return ok, beta0


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
