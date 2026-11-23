"""Exact counting of monochromatic 3-APs in a 2-colouring of [n] (FFT, integer-exact).

mono = sum over colour classes S of #{(a,d): d>=1, a, a+d, a+2d in S}
     = sum_S ( sum_y (1_S*1_S)(2y) 1_S(y) - |S| ) / 2
"""
import numpy as np


def mono3(col):
    """col: array of 0/1 of length n (index 0 <-> integer 1)."""
    n = len(col)
    tot = 0
    for c in (0, 1):
        s = (col == c).astype(np.int64)
        m = 1 << (2 * n).bit_length()
        F = np.fft.rfft(s.astype(np.float64), m)
        conv = np.rint(np.fft.irfft(F * F, m)[: 2 * n]).astype(np.int64)
        # index i in conv <-> sum of integers (i+2) ; 2y with y integer 1..n -> index 2y-2
        idx = 2 * np.arange(1, n + 1) - 2
        tot += (int(np.dot(conv[idx], s)) - int(s.sum())) // 2
    return tot


def block_colouring(n, lengths, unit, first=0):
    """lengths in units of n/unit, alternating colours starting with `first`."""
    col = np.zeros(n, dtype=np.int64)
    pos = 0
    c = first
    cuts = np.cumsum(lengths)
    assert cuts[-1] == unit
    start = 0
    for L, cut in zip(lengths, cuts):
        end = (cut * n) // unit
        col[start:end] = c
        start = end
        c ^= 1
    return col


if __name__ == "__main__":
    prs = [28, 6, 28, 37, 59, 116, 116, 59, 37, 28, 6, 28]
    for k in (10, 50, 200, 1000):
        n = 548 * k
        col = block_colouring(n, prs, 548)
        m = mono3(col)
        print(f"n={n}: mono/n^2 = {m / n**2:.7f}   (117/2192 = {117/2192:.7f}, random 1/16 = {1/16:.7f})")
