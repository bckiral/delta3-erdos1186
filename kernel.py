"""Continuous formulation of the 3-AP Ramsey multiplicity problem on [n].

For a colouring with local density g:[0,1]->[-1,1] (g = 2*red density - 1),
    #mono 3-APs / n^2  ->  1/16 + J(g)/4,   J(g) = ∬ H(x,y) g(x) g(y) dx dy,
    H = 1/4 + K_s/2,  K_s(x,y) = [2v-u<=1] + [2u-v>=0],  u=min(x,y), v=max(x,y).
(The 1/4 comes from the parity term, minimised when both parity classes have the
 same density.)  delta_3 = 1/16 + lambda*/4,  lambda* = min_g J(g).

Uniform grid with L cells.  For x = vector of cell averages of g:
    J(g) >= x^T A x / L^2 - E_L,
A_ij = L^2 ∬_{cell ij} H (exact), E_L = sum_cells ∬_cell |H - mean_cell(H)| (exact).
For step functions on the grid, J(g) = x^T A x / L^2 exactly.
"""
from fractions import Fraction as Fr
import numpy as np


def _int_clamp(lo_fun, hi_fun, a, h, ylo, yhi, exact):
    """∫_{x=a}^{a+h} max(0, min(hi(x), yhi) - max(lo(x), ylo)) dx, hi/lo affine (slope, icpt)."""
    (ms, mc) = hi_fun
    (ls, lc) = lo_fun
    F = (lambda t: Fr(t)) if exact else float
    a = F(a); h = F(h); ylo = F(ylo); yhi = F(yhi)
    ms, mc, ls, lc = F(ms), F(mc), F(ls), F(lc)
    pts = {a, a + h}
    def add_root(s, c, target):
        if s != 0:
            r = (target - c) / s
            if a < r < a + h:
                pts.add(r)
    add_root(ms, mc, yhi); add_root(ms, mc, ylo)
    add_root(ls, lc, ylo); add_root(ls, lc, yhi)
    if ms != ls:
        r = (lc - mc) / (ms - ls)
        if a < r < a + h:
            pts.add(r)
    pts = sorted(pts)
    def f(x):
        hi = min(ms * x + mc, yhi)
        lo = max(ls * x + lc, ylo)
        return hi - lo if hi > lo else F(0)
    tot = F(0)
    for p, q in zip(pts[:-1], pts[1:]):
        tot += (f(p) + f(q)) * (q - p) / 2  # integrand linear on [p,q]
    return tot


NEG_INF = (0, -10)   # lo(x) = -10 (below any y)
POS_INF = (0, 10)


def cell_areas(i, j, L, exact=False):
    """Areas, inside cell [i/L,(i+1)/L]x[j/L,(j+1)/L] restricted to x<y (cells with i<=j),
    of the four regions (R1&R2, R1 only, R2 only, neither), R1: y<=(1+x)/2, R2: y<=2x."""
    F = (lambda t: Fr(t)) if exact else float
    h = F(1) / L if exact else 1.0 / L
    a = F(i) * h
    ylo, yhi = F(j) * h, F(j + 1) * h
    lo = (1, 0) if i == j else NEG_INF          # y > x on the diagonal cell
    tot = _int_clamp(lo, POS_INF, a, h, ylo, yhi, exact)
    r1 = _int_clamp(lo, (F(1) / 2, F(1) / 2), a, h, ylo, yhi, exact)
    r2 = _int_clamp(lo, (2, 0), a, h, ylo, yhi, exact)
    # R1&R2: y <= min((1+x)/2, 2x); min of two affine functions has one kink at x=1/3
    k = F(1) / 3
    if a + h <= k:
        r12 = r2
    elif a >= k:
        r12 = r1
    else:
        r12 = _int_clamp(lo, (2, 0), a, k - a, ylo, yhi, exact) + \
              _int_clamp(lo, (F(1) / 2, F(1) / 2), k, a + h - k, ylo, yhi, exact)
    both = r12
    only1 = r1 - r12
    only2 = r2 - r12
    neither = tot - r1 - r2 + r12
    return both, only1, only2, neither, tot


def build(L, exact=False):
    """Returns A (L x L, cell-averaged H times L^2 ... i.e. A_ij = L^2 ∬ H) and E_L."""
    F = (lambda t: Fr(t)) if exact else float
    A = [[None] * L for _ in range(L)] if exact else np.zeros((L, L))
    E = F(0)
    vals = (F(5) / 4, F(3) / 4, F(3) / 4, F(1) / 4)  # both, only1, only2, neither
    LL = F(L * L)
    for i in range(L):
        for j in range(i, L):
            areas = cell_areas(i, j, L, exact)
            tot = areas[4]
            integ = sum(v * ar for v, ar in zip(vals, areas[:4]))
            mean = integ / tot
            dev = sum(abs(v - mean) * ar for v, ar in zip(vals, areas[:4]))
            mult = 2 if i == j else 1     # diagonal cell: mirror half; off-diagonal: counted for (i,j) and (j,i)
            aij = integ * mult * LL if i == j else integ * LL
            if exact:
                A[i][j] = aij; A[j][i] = aij
            else:
                A[i, j] = aij; A[j, i] = aij
            E += dev * 2  # (i,j) and (j,i) cells for i<j; diagonal: two mirror halves
    return A, E


def J_step(A, x):
    L = len(x)
    return float(x @ A @ x) / L**2


if __name__ == "__main__":
    import sys
    prs = [28, 6, 28, 37, 59, 116, 116, 59, 37, 28, 6, 28]
    L = 548 * 2
    A, E = build(L)
    x = np.zeros(L)
    pos = 0; c = 1.0
    for ln in prs:
        x[pos:pos + 2 * ln] = c
        pos += 2 * ln; c = -c
    lam = J_step(A, x)
    print(f"L={L}: J(PRS) = {lam:.8f}  (-5/137 = {-5/137:.8f});  beta = {1/16 + lam/4:.8f} (117/2192={117/2192:.8f});  E_L = {E:.6f}")
