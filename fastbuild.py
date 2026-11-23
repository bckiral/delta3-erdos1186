"""Fast exact construction of A (cell averages of H) and E_L for large L.

Uncut cells: H constant, value in {1/4, 3/4, 5/4}.  Cut cells (a line passes through the
interior): exact Fraction integration via kernel.cell_areas.  A is returned as an integer
matrix Az with A = Az / DEN (common denominator), plus the float matrix.
"""
from fractions import Fraction as Fr
from math import lcm
import numpy as np
from kernel import cell_areas


def classify(L):
    """For i<=j (x in B_i, y in B_j, x<y on/above diagonal):
    r1 status: +1 cell inside y<=(1+x)/2, -1 outside, 0 cut;  r2 status for y<=2x."""
    i = np.arange(L)[:, None].astype(np.int64); j = np.arange(L)[None, :].astype(np.int64)
    # coordinates scaled by L: x in [i,i+1], y in [j,j+1]
    # R1: 2y <= L + x. inside iff 2(j+1) <= L + i ; outside iff 2j >= L + i + 1
    r1 = np.where(2 * (j + 1) <= L + i, 1, np.where(2 * j >= L + i + 1, -1, 0))
    # R2: y <= 2x. inside iff j+1 <= 2i ; outside iff j >= 2(i+1)
    r2 = np.where(j + 1 <= 2 * i, 1, np.where(j >= 2 * (i + 1), -1, 0))
    return r1, r2


def build_fast(L):
    r1, r2 = classify(L)
    vals = {(1, 1): Fr(5, 4), (1, -1): Fr(3, 4), (-1, 1): Fr(3, 4), (-1, -1): Fr(1, 4)}
    Af = np.zeros((L, L))
    cut = {}
    E = Fr(0)
    V = (Fr(5, 4), Fr(3, 4), Fr(3, 4), Fr(1, 4))
    for i in range(L):
        for j in range(i, L):
            a, b = int(r1[i, j]), int(r2[i, j])
            if a != 0 and b != 0 and i != j:
                v = vals[(a, b)]
                Af[i, j] = Af[j, i] = float(v)
                continue
            # cut cell or diagonal cell: exact
            both, only1, only2, neither, tot = cell_areas(i, j, L, exact=True)
            areas = (both, only1, only2, neither)
            integ = sum(vv * ar for vv, ar in zip(V, areas))
            mean = integ / tot
            dev = sum(abs(vv - mean) * ar for vv, ar in zip(V, areas))
            aij = mean            # cell average of H (same for the diagonal half-cell)
            cut[(i, j)] = aij
            Af[i, j] = Af[j, i] = float(aij)
            E += 2 * dev
    # exact matrix as integers over a common denominator
    den = 4
    for v in cut.values():
        den = lcm(den, v.denominator)
    return Af, cut, E, den, r1, r2


def exact_entry(i, j, cut, r1, r2):
    if i > j:
        i, j = j, i
    if (i, j) in cut:
        return cut[(i, j)]
    a, b = int(r1[i, j]), int(r2[i, j])
    return {(1, 1): Fr(5, 4), (1, -1): Fr(3, 4), (-1, 1): Fr(3, 4), (-1, -1): Fr(1, 4)}[(a, b)]


if __name__ == "__main__":
    import sys, time
    from kernel import build
    L = int(sys.argv[1])
    t = time.time()
    Af, cut, E, den, r1, r2 = build_fast(L)
    print(f"L={L}: {len(cut)} exact cells, common denominator {den}, E_L = {float(E):.6f} [{time.time()-t:.1f}s]")
    if L <= 200:
        A2, E2 = build(L, exact=True)
        err = max(abs(Fr(A2[i][j]) - exact_entry(i, j, cut, r1, r2)) for i in range(L) for j in range(L))
        print("  cross-check vs slow exact build: max |diff| =", err, " E diff =", E2 - E)
