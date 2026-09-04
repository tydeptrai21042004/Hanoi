# DCT–QR Pairwise Coset-Optimized Gain-Normalized QIM

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

## 1. Active method

The public `dct_qr` proposal is **QR-conditioned Pairwise Coset-Optimized QIM**.
It preserves the former differential DCT carrier, three QR-conditioned QIM
steps, guard margins, integer-lattice closure, pilots, synchronization, and
carrier-specific canonical-QR gain normalization. The replaced component is the
payload-to-QIM parity assignment.

The implementation is in:

- `src/qr64_certified/proposals/method.py`;
- `configs/dct_qr_after_abc.json`;
- `scripts/validate_dct_qr_pairwise_coset.py`;
- `tests/test_dct_qr_pairwise_coset.py`.

## 2. Carrier and QR-conditioned QIM levels

For RGB pixel \(x=(R,G,B)^T\), define

\[
F(x)=0.299R+0.587G+0.114B+
\eta\left(R-\frac{G+B}{2}\right),\qquad \eta=0.07.
\]

For each \(8\times8\) block \(b\), compute the orthonormal DCT \(C_b\) and use

\[
v_b=\frac{C_b(0,1)-C_b(1,0)}{2}.
\]

QR reliability assigns

\[
\Delta_b\in\{18.25,13.25,10.25\},\qquad \rho_b=0.45\Delta_b.
\]

The step and guard margin are unchanged from the previous validated DCT–QR
configuration.

## 3. Safe parity sets

For encoded bit \(c\in\{0,1\}\), define

\[
\mathcal S_c(\Delta_b,\rho_b)=
\bigcup_{k\in\mathbb Z:\,k\bmod2=c}
\left[k\Delta_b-\frac{\Delta_b}{2}+\rho_b,
      k\Delta_b+\frac{\Delta_b}{2}-\rho_b\right].
\]

The minimum-distance projection is

\[
P_c(v_b)=\arg\min_{x\in\mathcal S_c(\Delta_b,\rho_b)}|x-v_b|^2.
\]

## 4. New embedding law

Let \(u_b\) be the Arnold-scrambled watermark bit. Sort payload blocks by QR
reliability and create consecutive groups \(G_j\) of two blocks. For each pair,
choose one shared binary coset label:

\[
\boxed{
s_j^*=\arg\min_{s\in\{0,1\}}
\sum_{b\in G_j}|P_{u_b\oplus s}(v_b)-v_b|^2.
}
\]

Embed

\[
\boxed{c_b=u_b\oplus s_j^*,\qquad v_b^*=P_{c_b}(v_b).}
\]

The existing minimum-norm RGB update and integer-lattice closure then enforce
the selected parity and guard margin in the final `uint8` image.

## 5. Extraction law

After synchronization and canonical \(r_{11}\)-based gain correction,

\[
\widetilde v_b=\frac{v_b^{(a)}}{\widehat\alpha_b^{\gamma}},
\qquad \gamma=0.90,
\]

extract

\[
\widehat c_b=
\operatorname{round}(\widetilde v_b/\Delta_b)\bmod2,
\qquad
\boxed{\widehat u_b=\widehat c_b\oplus s_j^*.}
\]

The implementation stores a packed payload flip mask in the key. The mask is
4096 bits (512 raw bytes); it represents 2048 independent pair labels without
storing the original host.

## 6. Mathematical guarantees

### Proposition 1: non-increasing projection distortion

For pair \(G_j\),

\[
D_j^{\mathrm{new}}=
\min_{s\in\{0,1\}}
\sum_{b\in G_j}|P_{u_b\oplus s}(v_b)-v_b|^2.
\]

The old mapping is the feasible case \(s=0\). Therefore

\[
\boxed{D_j^{\mathrm{new}}\le D_j^{\mathrm{old}}},
\qquad
\boxed{D^{\mathrm{new}}\le D^{\mathrm{old}}}.
\]

### Proposition 2: inverse labeling preserves the physical error event

Since \(c_b=u_b\oplus s_j^*\) and
\(\widehat u_b=\widehat c_b\oplus s_j^*\),

\[
\boxed{
\mathbf1[\widehat u_b\ne u_b]
=
\mathbf1[\widehat c_b\ne c_b].
}
\]

Thus XOR inversion introduces no additional bit error.

### Proposition 3: unchanged sufficient QIM margin

Because \(\Delta_b\) and \(\rho_b\) are unchanged, the previous sufficient
condition remains valid:

\[
|e_b|<\rho_b
\quad\Longrightarrow\quad
\widehat c_b=c_b
\quad\Longrightarrow\quad
\widehat u_b=u_b.
\]

## 7. Validated result

Protocol: 13 RGB hosts at \(512\times512\), one \(64\times64\) watermark, and
15 deterministic moderate attacks.

| Metric | Previous carrier-`r11` DCT–QR | Active pairwise-coset DCT–QR | Change |
|---|---:|---:|---:|
| Mean PSNR | 47.250421 dB | **50.399139 dB** | **+3.148718 dB** |
| Clean NC | 1.000000 | **1.000000** | 0 |
| Mean attacked NC | 0.997867 | **0.998028** | +0.000161 |
| Mean 10th-percentile NC | 0.993108 | **0.993362** | +0.000254 |
| Mean worst NC/host | 0.986162 | **0.988097** | +0.001935 |
| Global worst NC | 0.964654 | **0.964723** | +0.000069 |

The PSNR improvement comes from exact coset selection, not from reducing the
QIM step or guard margin.

## 8. Scope and limitation

The method remains blind with respect to the original host, but it is
key-assisted. The paper must disclose the packed 512-byte flip mask, the
existing gain-reference vector, and all other image-dependent side information.
The current benchmark uses one watermark and one seed; a publication study still
needs multiple watermarks, seeds, held-out parameter selection, statistical
testing, and stronger geometric/security attacks.
