"""Independent verification of the delta_3 lower-bound certificate.

Independent code paths from the certificate generator:
 * cell averages of H and the discretisation error E are recomputed by exact polygon
   clipping (Sutherland-Hodgman + shoelace, Fractions), not by the integration routine;
 * M' is rebuilt from the stored integer data only;
 * PSD is checked with a scipy Cholesky, a different scaling, and an exact flint residual.
Plus sanity checks of the discrete identity on small n.
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
import scipy.linalg as sla
import flint

# ---------- exact geometry ----------

def clip(poly, a, b, c):
    """keep points with a*x + b*y <= c (closed half-plane)."""
    out = []
    n = len(poly)
    for k in range(n):
        P, Q = poly[k], poly[(k + 1) % n]
        fp = a * P[0] + b * P[1] - c
        fq = a * Q[0] + b * Q[1] - c
        if fp <= 0:
            out.append(P)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return out


def area(poly):
    if len(poly) < 3:
        return Fr(0)
    s = Fr(0)
    for k in range(len(poly)):
        x1, y1 = poly[k]; x2, y2 = poly[(k + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def pieces(poly):
    """poly lies in {x <= y}.  H = 1/4 + 1/2[2y - x <= 1] + 1/2[2x - y >= 0] (u=x, v=y)."""
    res = []
    # R1: -x + 2y <= 1 ; not R1: x - 2y <= -1 ; R2: -2x + y <= 0 ; not R2: 2x - y <= 0
    for r1 in (True, False):
        p1 = clip(poly, -1, 2, 1) if r1 else clip(poly, 1, -2, -1)
        for r2 in (True, False):
            p2 = clip(p1, -2, 1, 0) if r2 else clip(p1, 2, -1, 0)
            val = Fr(1, 4) + (Fr(1, 2) if r1 else 0) + (Fr(1, 2) if r2 else 0)
            res.append((val, area(p2)))
    return res


def cell_exact(i, j, L):
    """average of H on cell B_i x B_j (any i,j) and integral of |H - avg|."""
    h = Fr(1, L)
    x0, y0 = i * h, j * h
    if i < j:
        polys = [([(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)], False)]
    elif i > j:
        # swap roles: H symmetric; cell maps to B_j x B_i under (x,y)->(y,x)
        return cell_exact(j, i, L)
    else:
        # diagonal cell: upper triangle (x<=y) and its mirror (same distribution)
        polys = [([(x0, y0), (x0 + h, y0 + h), (x0, y0 + h)], True)]
    pcs = []
    for poly, mirror in polys:
        for val, ar in pieces(poly):
            pcs.append((val, ar * (2 if mirror else 1)))
    tot = sum(ar for _, ar in pcs)
    assert tot == h * h
    mean = sum(v * ar for v, ar in pcs) / tot
    dev = sum(abs(v - mean) * ar for v, ar in pcs)
    return mean, dev


def build_indep(L):
    i = np.arange(L)[:, None]; j = np.arange(L)[None, :]
    # corners (scaled by L): x in {i, i+1}, y in {j, j+1}; only for i<j cells
    def side(fx):  # fx(x,y) -> sign at four corners; uncut iff all >=0 or all <=0
        vals = [fx(i + a, j + b) for a in (0, 1) for b in (0, 1)]
        V = np.stack(vals)
        return np.where((V <= 0).all(0), 1, np.where((V >= 0).all(0), -1, 0))
    s1 = side(lambda x, y: 2 * y - x - L)      # R1 iff <= 0
    s2 = side(lambda x, y: y - 2 * x)          # R2 iff <= 0
    A8 = np.zeros((L, L), dtype=np.int64)
    E = Fr(0)
    nexact = 0
    for a in range(L):
        for b in range(a, L):
            if a < b and s1[a, b] != 0 and s2[a, b] != 0:
                v = Fr(1, 4) + (Fr(1, 2) if s1[a, b] == 1 else 0) + (Fr(1, 2) if s2[a, b] == 1 else 0)
                dev = Fr(0)
            else:
                v, dev = cell_exact(a, b, L)
                nexact += 1
            v8 = v * 8
            assert v8.denominator == 1, (a, b, v)
            A8[a, b] = A8[b, a] = int(v8)
            E += dev * (1 if a == b else 2)
    return A8, E, nexact


# ---------- certificate check ----------

def check(cert_file, txt_file, tau_rel=2e-10, RB=38):
    t0 = time.time()
    Z = np.load(cert_file)
    cuts, nu_i, dp_i, K, Lf, Lc = Z["cuts"], Z["nu_int"], Z["dp_int"], int(Z["K"]), int(Z["Lf"]), int(Z["Lc"])
    m = Lf // Lc
    assert Lf == m * Lc
    assert (nu_i > 0).all() and (dp_i >= 0).all()
    signs = {0: (1, 1, 1), 1: (1, -1, -1), 2: (-1, 1, -1), 3: (-1, -1, 1)}
    for s in signs.values():
        assert s[0] * s[1] * s[2] == 1
    assert (cuts[:, 0] < cuts[:, 1]).all() and (cuts[:, 1] < cuts[:, 2]).all() and (cuts[:, 2] < Lc).all()
    A8, E, nexact = build_indep(Lf)
    print(f"  independent exact A and E built ({nexact} exact cells), E = {float(E):.9f} [{time.time()-t0:.0f}s]", flush=True)
    # C2 = 2^(K+1) * C(nu), C(nu) = sum nu_t T_t, T_t entries s/2
    C2 = np.zeros((Lc, Lc), dtype=object)
    C2[:] = 0
    for (a, b, c, s), nv in zip(cuts.tolist(), nu_i.tolist()):
        s1, s2, s3 = signs[s]
        C2[a, b] += nv * s1; C2[b, a] += nv * s1
        C2[b, c] += nv * s2; C2[c, b] += nv * s2
        C2[a, c] += nv * s3; C2[c, a] += nv * s3
    C2 = C2.astype(np.int64)
    blk = np.arange(Lf) // m
    Sm = A8 * (2 ** (K + 1) // 8) - Lc * Lc * C2[np.ix_(blk, blk)]
    Sm[np.diag_indices(Lf)] += 2 * dp_i
    assert (Sm == Sm.T).all()
    Mf = Sm.astype(float) / 2 ** (K + 1)
    tau = tau_rel * float(np.abs(Mf).max())
    Lchol = sla.cholesky(Mf - tau * np.eye(Lf), lower=True)       # Mf - tau I ~ L L^T
    Lz = flint.fmpz_mat([[int(v) for v in row] for row in np.rint(Lchol * 2 ** RB).astype(np.int64)])
    LLt = Lz * Lz.transpose()
    tau_i = int(np.floor(tau * 2 ** (K + 1)))
    sh = 2 * RB - (K + 1)
    Sz = flint.fmpz_mat([[int(v) for v in row] for row in (Sm - tau_i * np.eye(Lf, dtype=np.int64))])
    if sh >= 0:
        N = Sz * (flint.fmpz(2) ** sh) - LLt
        bound = tau_i * 2 ** sh
    else:
        N = Sz - LLt * 0  # not used
        raise SystemExit("choose RB with 2RB >= K+1")
    ent = N.entries()
    worst = max(sum(abs(int(x)) for x in ent[r * Lf:(r + 1) * Lf]) for r in range(Lf))
    ok = worst <= bound
    lam0 = -Fr(int(dp_i.sum()), 2 ** K) / (Lf * Lf) - Fr(int(nu_i.sum()), 2 ** K) - E
    beta0 = Fr(1, 16) + lam0 / 4
    claimed = None
    for line in open(txt_file):
        if line.startswith("beta0="):
            claimed = Fr(line.split("=", 1)[1].strip())
    print(f"INDEPENDENT CHECK Lf={Lf} Lc={Lc}: PSD {'OK' if ok else 'FAIL'} (residual/bound {worst/bound:.2e}); "
          f"delta_3 >= {float(beta0):.8f}; matches certificate file: {beta0 == claimed}  [{time.time()-t0:.0f}s]", flush=True)
    return ok and beta0 == claimed, beta0


def sanity(n=300, seed=1):
    """brute-force check of 4M = N3 + sum_{u<v} w(u,v) f_u f_v on random colourings."""
    rng = np.random.default_rng(seed)
    for trial in range(3):
        f = rng.choice([-1, 1], n)
        M = 0; N3 = 0
        for a in range(1, n + 1):
            for d in range(1, (n - a) // 2 + 1):
                N3 += 1
                if f[a - 1] == f[a + d - 1] == f[a + 2 * d - 1]:
                    M += 1
        Q = 0
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                w = (2 * v - u <= n) + (2 * u - v >= 1) + ((v - u) % 2 == 0)
                Q += w * f[u - 1] * f[v - 1]
        assert 4 * M == N3 + Q, (M, N3, Q)
    print(f"  identity 4M = N3 + sum w f f verified on 3 random colourings of [{n}]")


if __name__ == "__main__":
    sanity()
    ok, b = check(sys.argv[1], sys.argv[2])
    print("RESULT:", "VERIFIED" if ok else "NOT VERIFIED", float(b))
