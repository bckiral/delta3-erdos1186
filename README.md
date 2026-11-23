# A certified lower bound for monochromatic 3-term progressions (Erdős Problem #1186, k = 3)

**Theorem.** Every 2-colouring of {1, ..., n} contains at least (β₀ − o(1)) n² monochromatic
3-term arithmetic progressions, where

    β₀ = 8801285356716686183 / 166020696663385964544 = 0.05301318169...

Previous best rigorous lower bound: 1675/32768 ≈ 0.0511169 (Parrilo–Robertson–Saracino, 2008).
Best known upper bound: 117/2192 ≈ 0.0533759 (same paper).

## Verify (about 15 seconds)

Requirements: Python 3, numpy, scipy, python-flint (`pip install numpy scipy python-flint`).

    python3 verify_indep.py cert_hybrid_1536_96.npz cert_hybrid_1536_96.txt

Expected last line:

    RESULT: VERIFIED 0.05301318169120605

`verify_indep.py` is standalone. It
1. brute-force checks the identity 4M(f) = N3 + Σ_{u<v} w(u,v) f_u f_v on random colourings;
2. recomputes the cell averages A and the discretisation error E_L by exact polygon clipping
   (Fractions);
3. rebuilds M' = A + diag(d') − Lc² · kron(C(ν), J) from the stored integers and proves M' ⪰ 0 by an
   exact integer residual test (Cholesky factor rounded to integers, residual computed with FLINT,
   Gershgorin bound);
4. recomputes λ₀ = −Σd − Σν − E_L and β₀ = 1/16 + λ₀/4 exactly and compares with the .txt file.

## Logic of the proof (see the write-up for details)

1. Exact identity → M(f)/n² ≥ 1/16 + J(g)/4 − O(1/n), with J(g) = ∬ H g g on [0,1]²,
   H = 1/4 + (1/2)(1[2max−min ≤ 1] + 1[2min−max ≥ 0]).
2. Grid with L blocks: J(g) ≥ xᵀAx/L² − E_L (A = cell means of H, exact; E_L exact).
3. Dual certificate: if M = A/Lf² + diag(d) − Pᵀ(Σ ν_t T_t)P ⪰ 0 with d, ν ≥ 0 (T_t = coarse triangle
   inequalities, valid on the cube), then xᵀAx/Lf² ≥ −Σd − Σν on [−1,1]^Lf.
4. Lf = 1536 (fine grid), Lc = 96 (triangle grid), 40,732 positive triangle multipliers.

## Files

| File | Purpose |
| --- | --- |
| `cert_hybrid_1536_96.npz` | certificate: triangles (i,j,k,sign), integer ν and d' (scale 2^-44) |
| `cert_hybrid_1536_96.txt` | exact rationals: λ₀, β₀, E_L, Σd', Σν |
| `cert_hybrid_384_96.*` | a smaller certificate (β₀ ≥ 0.0526611) |
| `cert_shor_2048.txt` | Shor-only certificate (β₀ ≥ 0.0520514) |
| `verify_indep.py` | independent checker (no shared geometry code) |
| `kernel.py`, `fastbuild.py` | exact construction of A and E_L |
| `tri2.py` | Shor + triangle SDP at L = 96 with cutting planes (SCS via cvxpy) |
| `tri2_cuts_96.npy`, `tri2_mu_96.npy` | its triangle list and dual multipliers |
| `lag.py`, `certify_hybrid.py` | fine-grid Shor solve and exact certificate generation |
| `certify2.py`, `shor.py` | Shor-only certificate |
| `count.py` | exact FFT count of monochromatic 3-APs (checks the PRS colouring) |

Regenerate the main certificate: `python3 certify_hybrid.py 1536 96 tri` (needs the two .npy files).

Environment used: Python 3.13, numpy 2.5.3, scipy 1.18.1, cvxpy 1.9.3 (SCS), python-flint 0.9.0.

## AI disclosure

Produced with extensive assistance from Claude (Anthropic): literature search, derivation, code and
computation. Correctness rests on the machine-checkable certificate above.
