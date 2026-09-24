# DCT-QR Theory-Grounded Variant (`dct_qr_theory`)

This variant is added beside the original `dct_qr`; the original implementation is unchanged. The purpose is narrow: **replace only the empirically fixed red-item decisions** while retaining the green/yellow DCT-QR architecture.

## Retained architecture

The method still uses BT.601 luminance plus an opponent term, 8x8 orthonormal DCT-II, the differential carrier

\[
v_b=\frac{C_{01}-C_{10}}{2},
\]

the 4x4 low-frequency analysis matrix with DC replacement, diagonal lifting, canonical QR, the separate determinant/Frobenius `beta`, QR diagonal balance and upper-triangular coupling descriptors, adaptive stronger QIM for weaker blocks, minimum-norm RGB updates, integer-lattice closure, Arnold scrambling, pilots, QR-`r11` gain estimation, distance-based QIM confidence, and a four-neighbour Ising/ICM prior.

## Red-item replacements

### 1. Opponent coefficient

Instead of the tuned `eta=0.07`, let

\[
L=(0.299,0.587,0.114)^T,\qquad O=(1,-1/2,-1/2)^T.
\]

Choose the nonzero solution of

\[
\|L+\eta O\|_2=\|L\|_2,
\]

namely

\[
\eta^*=-\frac{2L^TO}{O^TO}=0.068666\ldots.
\]

This keeps the opponent term while preserving the Euclidean sensitivity norm used by the minimum-norm RGB update.

### 2. Diagonal lift and nonsingularity

Let `A_b^0` be the unlifted 4x4 analysis matrix and choose

\[
\lambda=\operatorname{nextafter}\!\left(\max_b\|A_b^0\|_F,+\infty\right).
\]

Since \(\|A\|_2\le\|A\|_F\), each block satisfies \(\|A_b^0/\lambda\|_2<1\). Therefore

\[
A_b=A_b^0+\lambda I=\lambda(I+A_b^0/\lambda)
\]

is nonsingular by the Neumann lemma. No determinant epsilon threshold is required for the theoretical construction.

### 3. No weighted reliability fusion

The method does **not** form

\[
\log(1+\beta_b)+2B_b+0.2(1-C_b)
\]

and does not apply 5--95 percentile normalization. The three descriptors remain separate. Adaptive strength uses only the theorem-backed nonsingularity indicator

\[
\beta_b=\frac{|\det A_b|}{\|A_b\|_F^3}.
\]

### 4. Continuous adaptive step instead of 20/40/40 classes

The reference QIM step follows directly from the 8x8 differential-carrier geometry. Two orthonormal DCT atoms give perturbation norm \(\sqrt2|\delta|\), hence block RMS \(\sqrt2|\delta|/8\). Therefore

\[
\Delta_0=\frac{8}{\sqrt2}.
\]

For payload blocks, with geometric mean \(G_\beta\), use

\[
\Delta_b=\Delta_0\frac{G_\beta}{\beta_b}.
\]

Thus smaller `beta` receives larger spacing without thresholds, ranks, fixed 20/40/40 fractions, or the fixed values 18.25/13.25/10.25.

### 5. Center-QIM instead of `rho=0.45 Delta`

The continuous projection is to the nearest center of the requested parity coset. Before integer rounding its decision-boundary margin is exactly

\[
\Delta_b/2.
\]

The integer-lattice closure then restores the requested parity if bounded uint8 rounding changes it.

### 6. Global coset instead of reliability-sorted pairs

Use one bit \(s\in\{0,1\}\) and select

\[
s^*=\arg\min_{s\in\{0,1\}}\sum_b
\left|P_{u_b\oplus s}(v_b)-v_b\right|^2.
\]

Because `s=0` is feasible, optimized projection energy cannot exceed the original labeling. There is no pair-size parameter and no reliability sorting.

### 7. Exact QR gain exponent

For canonical QR and a positive scalar gain \(A_b'=\alpha_bA_b\),

\[
r_{11,b}'=\alpha_b r_{11,b}.
\]

Therefore extraction uses

\[
\widetilde v_b=v_b/\widehat\alpha_b
\]

with exponent exactly `gamma=1` and no empirical gain clipping.

### 8. ICM without fitted lambda, diagonal weight, or iteration cap

The spatial prior uses only the four-neighbour grid. The coupling is degree-normalized by the maximum grid degree,

\[
\lambda=1/4,
\]

and updates are sequential until a fixed point. This removes `lambda_MAP=0.33`, the diagonal weight `0.35`, and the fixed 12-iteration cap.

## Extraction terminology

The method requires no original host image, but its key stores adaptive steps, a global coset bit, a QR gain reference, and schedules. It is therefore described as **host-blind, key-assisted extraction**.

## Reproducible ablation and common-20 protocol

The executable one-factor ablations, curated 20-attack main benchmark, and numerical illustrations are documented in `docs/DCT_QR_THEORY_EXPERIMENT_PROTOCOL.md`. Use that protocol for manuscript tables rather than mixing the legacy 15-attack suite with severe stress attacks.
