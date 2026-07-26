# DCT–Schur SP-SCQIM: corrected implementation, mathematics, and validation report

## 1. Executive conclusion

The former public `dct_schur_rescue` path was not an independent Schur proposal. The source file contained two public `embed()` definitions and two public `extract()` definitions. Python silently retained only the last pair, which called the DCT–QR embedding and extraction engine. The advertised spectrum-preserving direct-Schur code was therefore shadowed.

This revision replaces that path with an independent method:

> **SP-SCQIM — Spectrum-Preserving Schur Coupling Quantization Index Modulation**

The corrected method has its own carrier, embedding law, key, decoder, candidate-selection rule, configuration, tests, and benchmark. It does not call the DCT–QR proposal engine.

The mathematical separation is achieved. The requested simultaneous performance dominance is **not** achieved on the complete 13-host gate:

| Method | Mean PSNR | Clean NC | Mean attacked NC | Mean worst NC/host | Global worst NC |
|---|---:|---:|---:|---:|---:|
| Previous Schur-gain DCT-QIM | 48.014086 | 1.000000 | **0.996058** | **0.975353** | **0.923018** |
| New SP-SCQIM, step 8.5 | **48.650100** | 1.000000 | 0.992847 | 0.933349 | 0.710975 |
| New SP-SCQIM, step 9.0 | 48.173152 | 1.000000 | 0.993814 | 0.942074 | 0.740984 |

Step 9.0 is the default because it remains above the old mean PSNR while giving the best tested robustness in the admissible PSNR region. Nevertheless,

\[
0.993814 < 0.996058,
\]

so the code must not claim that SP-SCQIM already improves both metrics over the previous hybrid.

## 2. Corrected repository architecture

### Public method

```text
src/qr64_certified/proposals/schur_coupling_qim.py
```

contains the only active Schur implementation.

### Compatibility wrapper

```text
src/qr64_certified/proposals/direct_schur_rescue.py
```

is now a thin compatibility import. It no longer contains duplicate public functions.

### Historical implementation

```text
src/qr64_certified/proposals/direct_schur_rescue_legacy.py
```

retains the former implementation for audit and negative-result comparison. It is not used by the public registry.

### Active configuration

```text
configs/dct_schur_sp_scqim.json
```

### Reproducible validation

```bash
python scripts/benchmark_schur_sp_scqim.py \
  --step 9.0 \
  --output results/schur_sp_scqim/reproduced.json
```

## 3. Domain construction

For each 8×8 opponent-field DCT block, define the six-dimensional strict-upper Schur coupling vector

\[
n_b=
\begin{bmatrix}
t_{12}&t_{13}&t_{14}&t_{23}&t_{24}&t_{34}
\end{bmatrix}^{\!T}.
\]

The six coordinates are taken from the fixed DCT positions

\[
(0,1),(1,0),(0,2),(1,1),(2,0),(1,2).
\]

Four disjoint coefficients define the unchanged diagonal of a virtual upper-triangular Schur matrix:

\[
T_b=
\begin{bmatrix}
\lambda_{b,1} & t_{12} & t_{13} & t_{14}\\
0 & \lambda_{b,2} & t_{23} & t_{24}\\
0 & 0 & \lambda_{b,3} & t_{34}\\
0 & 0 & 0 & \lambda_{b,4}
\end{bmatrix}.
\]

A positive virtual lift is applied only when checking nonsingularity. It is not embedded into the image.

## 4. Three orthogonal Schur coupling carriers

The method uses

\[
H=
\begin{bmatrix}
h_1^T\\h_2^T\\h_3^T
\end{bmatrix},
\qquad HH^T=I_3,
\]

with

\[
h_1=\frac{1}{\sqrt2}(1,-1,0,0,0,0)^T,
\]

\[
h_2=\frac12(1,1,-1,-1,0,0)^T,
\]

\[
h_3=\frac{1}{\sqrt{12}}(1,1,1,1,-2,-2)^T.
\]

The three scalar carriers are

\[
c_{b,k}=h_k^Tn_b,
\qquad k=1,2,3.
\]

Each carrier receives a differently interleaved copy of the scrambled 4,096-bit payload. Thus one payload bit has three observations located in different blocks and coupling directions.

## 5. Explicit embedding law

For a carrier value \(c\), bit \(w\in\{0,1\}\), and step \(\Delta\), define the parity lattice

\[
\mathcal L_w(\Delta)=
\{q\Delta:q\in\mathbb Z,\ q\bmod2=w\}.
\]

The target is

\[
t=\operatorname*{argmin}_{z\in\mathcal L_w(\Delta)}|z-c|.
\]

For one block, collect all three targets in

\[
t_b=(t_{b,1},t_{b,2},t_{b,3})^T.
\]

The embedding update is

\[
\boxed{
 n_b^*=n_b+H^T(t_b-Hn_b).
}
\]

This is the implemented `luật nhúng`. It is not the DCT–QR carrier

\[
(C_{01}-C_{10})/2,
\]

and it does not call the QR proposal's QIM engine.

## 6. Mathematical contributions

### Theorem 1 — exact satisfaction of all coupling constraints

Because \(HH^T=I_3\),

\[
Hn_b^*
=Hn_b+HH^T(t_b-Hn_b)
=t_b.
\]

Therefore every floating-point coupling reaches its selected parity lattice exactly.

The implementation tests the residual

\[
\|Hn_b^*-t_b\|_\infty.
\]

### Theorem 2 — minimum-Frobenius update

Consider

\[
\min_x\|x-n_b\|_2^2
\quad\text{subject to}\quad Hx=t_b.
\]

The Lagrangian is

\[
L(x,\mu)=\frac12\|x-n_b\|_2^2+\mu^T(Hx-t_b).
\]

Stationarity gives

\[
x=n_b-H^T\mu.
\]

Using \(Hx=t_b\) and \(HH^T=I_3\),

\[
\mu=Hn_b-t_b.
\]

Hence

\[
x=n_b+H^T(t_b-Hn_b)=n_b^*.
\]

Since the objective is strictly convex, this solution is unique. Because the six-vector is exactly the strict-upper part of \(T_b\), minimizing its Euclidean change is equivalent to minimizing the Frobenius change of the virtual Schur matrix.

### Theorem 3 — spectrum, trace, and determinant preservation

The update modifies only the strict-upper entries. The diagonal remains unchanged:

\[
\operatorname{diag}(T_b^*)=
\operatorname{diag}(T_b).
\]

For an upper-triangular matrix, the eigenvalues are its diagonal entries. Therefore

\[
\operatorname{spec}(T_b^*)=
\operatorname{spec}(T_b),
\]

\[
\operatorname{tr}(T_b^*)=
\operatorname{tr}(T_b),
\]

and

\[
\det(T_b^*)=
\prod_{i=1}^4\lambda_{b,i}
=
\det(T_b).
\]

The tests verify these identities numerically with tolerance \(10^{-8}\).

### Theorem 4 — clean parity decision condition

Let the post-rounding carrier be

\[
\widehat c_{b,k}=t_{b,k}+e_{b,k}.
\]

Nearest-integer parity decoding remains correct when

\[
|e_{b,k}|<\frac{\Delta}{2}.
\]

The implementation performs iterative uint8 closure: after inverse DCT, RGB projection, clipping, and rounding, it remeasures all three physical copies and reprojects any residual lattice errors.

## 7. Extraction

For each coupling,

\[
q_{b,k}=\operatorname{round}(c_{b,k}/\Delta),
\qquad
\widehat w_{b,k}=q_{b,k}\bmod2.
\]

The lattice confidence is

\[
\chi_{b,k}
=
\max\left(
0,
1-2\left|
\frac{c_{b,k}}{\Delta}-q_{b,k}
\right|
\right).
\]

The signed observation is

\[
e_{b,k}
=(2\widehat w_{b,k}-1)
\left(c_0+c_1\chi_{b,k}^{p}\right).
\]

After undoing each interleaver, the three observations are summed. A small Potts/ICM prior is applied after inverse Arnold permutation.

The method also evaluates a deterministic set of identity/sharpening candidates. Candidate selection uses only internal repetition agreement, evidence magnitude, and confidence; it does not use the original host or watermark.

## 8. Difference from the other proposals

| Component | DCT–QR | SP-SCQIM DCT–Schur | Spatial DetQR |
|---|---|---|---|
| Domain | DCT + QR-conditioned QIM | DCT strict-upper Schur coordinates | Spatial 2×2 QR residual |
| Payload carrier | one coefficient difference | three orthogonal Schur couplings | normalized spatial residual |
| Embedding law | parity projection/coset optimization | \(n^*=n+H^T(t-Hn)\) | minimum integer residual update |
| Principal invariant | QIM decision guard | spectrum, trace, determinant | strict nonsingularity \(\det A\ne0\) |
| Redundancy | carrier schedule | three interleaved Schur copies | spatial payload/pilot pattern |
| Decoder | QR gain-normalized QIM | Schur coupling vote fusion | determinant-safe residual sign |
| Public engine shared | — | **No** | No |

## 9. Test results

The revised suite reports:

```text
49 passed
```

Added tests cover:

1. exactly one public `embed()` and one public `extract()` definition;
2. no call to the DCT–QR public embedding/extraction engine;
3. orthonormality of \(H\);
4. exact floating-point spectrum, trace, and determinant preservation;
5. minimum-projection constraint residual;
6. clean 64×64 round trip;
7. active step parameter;
8. registry still exposing exactly three proposals.

Smoke test on Lenna:

| Proposal | PSNR | Clean NC |
|---|---:|---:|
| DCT–QR | 50.277718 | 1.000000 |
| SP-SCQIM DCT–Schur | 48.156553 | 1.000000 |
| Spatial DetQR | 55.872211 | 1.000000 |

## 10. Complete 13-host result

Protocol:

- 13 RGB hosts, 512×512;
- one 64×64 binary watermark;
- 4,096 payload bits;
- 15 deterministic moderate attacks;
- seed 2026;
- step 9.0.

Aggregate:

\[
\overline{\mathrm{PSNR}}=48.173152\ \mathrm{dB},
\]

\[
\mathrm{NC}_{clean}=1,
\]

\[
\overline{\mathrm{NC}}_{attacked}=0.993814,
\]

\[
\overline{\mathrm{NC}}_{worst/host}=0.942074,
\]

\[
\mathrm{NC}_{global\ worst}=0.740984.
\]

The difficult cases are:

| Attack | Mean NC | Minimum NC | Worst host |
|---|---:|---:|---|
| Median 3×3 | 0.947264 | 0.740984 | Baboon |
| Gaussian blur radius 1 | 0.978388 | 0.831636 | Baboon |
| JPEG Q70 | 0.992243 | 0.961951 | Girl |

The method is strong on noise, rotation-back, gamma, brightness, contrast, and resizing, but its current strict-upper carrier is not sufficiently stable under nonlinear median filtering on highly textured images.

## 11. Step sweep and promotion decision

| Step | Mean PSNR | Mean attacked NC | Decision |
|---:|---:|---:|---|
| 8.5 | 48.650100 | 0.992847 | best PSNR, weaker NC |
| 9.0 | 48.173152 | 0.993814 | selected default |

For Baboon alone:

| Step | PSNR | Mean NC | Worst NC |
|---:|---:|---:|---:|
| 9.0 | 48.179144 | 0.970343 | 0.740984 |
| 10.0 | 47.303989 | 0.976563 | 0.798869 |
| 10.5 | 46.882340 | 0.979505 | 0.814636 |
| 11.0 | 46.499622 | 0.979010 | 0.798544 |

Increasing the step cannot close the robustness gap without violating the desired PSNR behavior. The failure is therefore structural, not a simple hyperparameter issue.

## 12. Honest scientific status

### Achieved

- genuine carrier-level separation from DCT–QR and Spatial DetQR;
- explicit and independently implemented embedding law;
- minimum-Frobenius projection theorem;
- exact floating-point spectrum/trace/determinant invariance;
- clean NC 1 on all 13 hosts;
- mean PSNR above the former Schur method;
- deterministic full benchmark and independent tests.

### Not achieved

- higher aggregate attacked NC than the previous Schur-gain DCT-QIM method;
- median-filter robustness on highly textured hosts;
- a valid claim of universal superiority.

The scientifically correct label is:

> **Independent proposal with validated mathematical invariants and clean closure; robustness performance gate not yet passed.**

It should not be called a validated superior replacement until the NC gate is met on held-out hosts, watermark patterns, and seeds.

## 13. Next method-level research needed

The next improvement should change the carrier model rather than only increasing \(\Delta\). Promising directions are:

1. learn a fixed Schur coupling basis from the generalized eigenvectors of clean-versus-attack covariance on a training split;
2. use a differentiable median-filter surrogate when designing \(H\), then freeze \(H\) before testing;
3. replace repetition-3 with a rate-1/3 structured binary code across the three Schur channels;
4. add a second disjoint Schur subspace activated only when internal copy disagreement exceeds a threshold;
5. evaluate on held-out hosts so attack-aware basis design does not leak test information.

These changes can preserve the same spectrum-preserving projection theorem while directly addressing the observed structural failure.
