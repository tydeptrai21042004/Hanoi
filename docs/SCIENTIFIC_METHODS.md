# Scientific formulation of the proposal methods

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

This document describes each method as a sequence of mathematical components. Each component has four items:

1. **Object:** the quantity being modeled;
2. **Principle:** the mathematical reason for using it;
3. **Decision rule:** the equation used by the method;
4. **Evidence:** the validation condition or measured effect.

The purpose is to keep the proposals scientific and falsifiable. Implementation details such as thread counts, file formats, and execution shortcuts are deliberately excluded.

---

# 1. DCT-QR reliability-conditioned watermarking

## Scientific question

Can the local stability of a QR-factorized image matrix be used to allocate watermark strength so that weak blocks receive more protection and stable blocks receive less distortion?

## Component 1: opponent-color observation field

### Object

For RGB values, define

\[
F=0.299R+0.587G+0.114B+\eta\left(R-\frac{G+B}{2}\right),
\qquad \eta=0.07.
\]

### Principle

The luminance term carries most visible structure. The opponent term adds a small chromatic direction that is not identical to luminance. This gives the carrier two sources of image variation while keeping the perturbation interpretable.

### Decision rule

All DCT coefficients used by the method are calculated from \(F\), not independently from the three channels.

### Evidence

The same field is used before and after the proposed change, so the before/after comparison isolates the contribution of QR-conditioned strength allocation.

## Component 2: differential DCT carrier

### Object

For an \(8\times8\) block, let \(C(u,v)\) denote its orthonormal DCT coefficient. The payload carrier is

\[
u=C(0,1)-C(1,0).
\]

### Principle

A difference of two low-frequency AC coefficients suppresses the DC level and is less dependent on uniform brightness. Because the DCT basis is orthonormal, the squared coefficient displacement has a direct relation to spatial-domain energy.

## Component 3: binary QIM classes

### Object

A bit \(b\in\{0,1\}\) is represented by one of two interlaced lattices:

\[
\mathcal Q_b(\Delta)=\{(2k+b)\Delta:k\in\mathbb Z\}.
\]

### Principle

The nearest valid lattice point minimizes carrier displacement subject to exact binary separability. The protected radius \(\rho<\Delta/2\) creates a guard interval around the decision boundary.

### Decision rule

\[
u'=\arg\min_{q\in\mathcal Q_b(\Delta)}|u-q|
\]

with integer-domain closure so that the final uint8 image remains in the intended class.

## Component 4: QR stability certificate

### Object

From low-frequency AC coefficients, form a lifted matrix \(A_b\) for each block and compute

\[
A_b=Q_bR_b.
\]

The certificate combines:

\[
|\det A_b|,
\qquad
\frac{\min_i |r_{ii}|}{\max_i |r_{ii}|+\varepsilon},
\qquad
\frac{\|R_b-\operatorname{diag}(R_b)\|_F}{\|R_b\|_F+\varepsilon}.
\]

These quantities are mapped to a reliability value \(r_b\in[0,1]\).

### Principle

A small determinant or strongly imbalanced triangular factor indicates that the local matrix is close to degeneracy. Small perturbations can then cause larger relative changes in its derived quantities. The certificate is therefore a local stability indicator, not a second watermark carrier.

## Component 5: reliability-conditioned strength allocation

### Object

Blocks are ordered by \(r_b\). The validated three-level rule is

\[
\Delta_b=
\begin{cases}
1.28125\Delta,& r_b\text{ in the weakest }20\%,\\
\Delta,& r_b\text{ in the middle }40\%,\\
0.8125\Delta,& r_b\text{ in the strongest }40\%.
\end{cases}
\]

For the validated reference \(\Delta=16\), the levels are \(20.5,16,13\).

### Principle

The rule approximates the constrained allocation problem

\[
\min_{\{\Delta_b\}}\sum_b D_b(\Delta_b)
\quad\text{subject to}\quad
\sum_b P_b(\text{error}\mid r_b,\Delta_b)\leq \epsilon.
\]

Stable blocks require less separation to reach the same error probability, while unstable blocks require more. A uniform step cannot exploit this difference.

### Measured evidence

Across 13 hosts and 15 attacks:

| Metric | Uniform reference | QR-conditioned | Change |
|---|---:|---:|---:|
| Mean PSNR | 45.635875 dB | 45.684599 dB | +0.048724 dB |
| Clean NC | 0.999981 | 1.000000 | +0.000019 |
| Mean NC | 0.990520 | 0.990990 | +0.000471 |
| Mean worst-attack NC | 0.959743 | 0.962965 | +0.003222 |

The gain is small but simultaneous, so it is accepted.

## Component 6: pilot and geometric null hypothesis

### Object

Let \(J(T)\) combine pilot agreement and certificate consistency under candidate transform \(T\). Identity is \(I\).

### Decision rule

\[
T^*=\arg\max_T J(T),
\qquad
T^*\text{ is accepted only if }J(T^*)-J(I)>\tau.
\]

### Principle

Identity is the null hypothesis. Requiring a positive improvement prevents unnecessary interpolation when no geometric attack is present.

## Component 7: spatial prior for uncertain bits

### Object

For binary labels \(s_i\in\{-1,+1\}\), the decoding energy is

\[
E(s)=-\sum_i L_i s_i-\lambda\sum_{(i,j)\in\mathcal N}s_is_j,
\]

where \(L_i\) is the signed local evidence.

### Principle

The first term preserves observed data. The second expresses the limited assumption that neighboring logo pixels are more likely to agree than independent noise would suggest. Exact high-confidence clean bits bypass this prior.

## Defensible novelty

The novelty is not the separate use of DCT, QIM, or QR. It is the active coupling:

> QR reliability controls local QIM separation, contributes to extraction evidence, and gates geometric correction under one stability model.

---

# 2. DCT-Schur invariant rescue hypothesis

## Scientific status

This method is retained as an **exploratory hypothesis**, not a validated improvement. Its secondary channel must satisfy

\[
\operatorname{Accuracy}_{\text{Schur, clean}}\geq0.99
\]

before fusion can be credited as a reliable rescue mechanism.

## Scientific question

Can a spectrum-preserving change in matrix non-normality provide information statistically independent from a primary DCT-QIM decision?

## Component 1: controlled primary reference

The primary DCT-QIM model is kept fixed. Adaptive QR strength is disabled in this method so that any gain or loss can be attributed to the Schur component rather than to simultaneous changes in the primary channel.

## Component 2: real Schur representation

### Object

A matrix assembled from 16 low/mid-frequency coefficients is written as

\[
A=ZTZ^T,
\]

where \(T\) is real quasi-upper triangular. Decompose

\[
T=D+N,
\]

where \(D\) contains the \(1\times1\) and \(2\times2\) diagonal blocks and \(N\) is the strict block-upper part.

### Principle

The eigenvalues of a real Schur matrix are determined by its diagonal blocks. Holding \(D\) fixed while changing \(N\) preserves the spectrum in floating-point arithmetic.

## Component 3: departure from normality

### Object

The normalized Henrici departure is

\[
\nu(A)=
\frac{\sqrt{\|A\|_F^2-\sum_i|\lambda_i(A)|^2}}
{\sqrt{\sum_i|\lambda_i(A)|^2}+1}.
\]

### Principle

For fixed eigenvalues, changing the strict upper structure changes non-normality without changing the spectral values. This creates a secondary scalar degree of freedom.

## Component 4: invariant-preserving modulation

### Object

Scale only \(N\):

\[
T'=D+\alpha N,
\qquad
A'=ZT'Z^T.
\]

### Mathematical consequence

Because \(D\) is unchanged:

\[
\operatorname{spec}(A')=\operatorname{spec}(A),
\]

and therefore, in exact arithmetic,

\[
\operatorname{tr}(A')=\operatorname{tr}(A),
\qquad
\det(A')=\det(A).
\]

The parameter \(\alpha\) is selected so that \(\nu(A')\) belongs to the parity class of the rescue bit.

## Component 5: confidence-gated fusion

### Object

Let \(L_p\) be primary evidence and \(L_s\) Schur evidence. A scientifically justified rescue should act only when primary evidence is weak:

\[
L=L_p+g(|L_p|)L_s,
\qquad
g'(x)\leq0.
\]

### Principle

A secondary channel should not overwrite confident primary observations. It should reduce uncertainty only in an erasure-like region.

## Current evidence and rejection

The tested sparse joint-rescue variant produced:

| Metric | Before | Candidate | Change |
|---|---:|---:|---:|
| Mean PSNR | 46.937031 dB | 47.031285 dB | +0.094254 dB |
| Mean NC | 0.988850 | 0.988802 | -0.000048 |
| Worst-attack NC | 0.953918 | 0.957539 | +0.003621 |
| Selected clean Schur accuracy | — | 0.858173 | target 0.99 |

The candidate is rejected because the independent channel criterion is not met and average NC does not improve.

## Required scientific redesign

The next valid hypothesis is a constrained projection:

\[
\min_{\Delta c}\|\Delta c\|_2^2
\]

subject to

\[
\text{primary QIM condition},
\qquad
\text{Schur parity condition},
\qquad
\nabla u^T\Delta c=0.
\]

The last equation places the Schur update in the null space of the primary carrier to first order. This redesign is not claimed as validated in the package.

## Defensible novelty after successful validation

> Spectrum-preserving modulation of Schur non-normality, constrained to the primary-carrier null space and used only as independently validated erasure evidence.

---

# 3. Spatial channel-differential DetQR watermarking

## Scientific question

Can a spatial statistic have an exact, low-distortion update law while also supplying a synchronization signal in the same mathematical domain?

## Component 1: balanced spatial partitions

### Object

Each \(8\times8\) block is reduced to a \(4\times4\) cell-mean array. A balanced Hadamard pattern \(P\in\{-1,+1\}^{4\times4}\) defines two sets:

\[
S_+=\{i:P_i=+1\},
\qquad
S_-=\{i:P_i=-1\}.
\]

### Principle

Balanced partitions cancel the common block mean and measure structured contrast. Low-boundary patterns are less oscillatory and are generally less sensitive to smoothing and compression.

## Component 2: signed QR determinant

### Object

For channel \(c\), construct

\[
A_{c,p}=
\begin{bmatrix}
\mu_c(S_+) & 1\\
\mu_c(S_-) & 1
\end{bmatrix}
=Q_{c,p}R_{c,p}.
\]

The signed determinant is

\[
d_{c,p}=\det(Q_{c,p})r_{11}r_{22}
=\mu_c(S_+)-\mu_c(S_-).
\]

### Principle

The QR expression provides a nonsingularity certificate, while the equality to a mean difference gives an exact spatial interpretation.

## Component 3: channel-differential carrier

### Object

For two channels \(c_1,c_2\), define

\[
x=d_{c_1,p}-d_{c_2,p}.
\]

### Principle

Common illumination changes affecting both channels similarly are partly canceled. The sign of \(x\) represents the coded bit.

## Component 4: exact antisymmetric update

### Object

Add \(+saP\) to channel \(c_1\) and \(-saP\) to channel \(c_2\), where \(s\in\{-1,+1\}\) is the desired sign and \(a\in\mathbb Z_{\geq0}\).

### Mathematical consequence

For the selected partition,

\[
d_{c_1,p}'=d_{c_1,p}+2sa,
\qquad
d_{c_2,p}'=d_{c_2,p}-2sa,
\]

so

\[
x'=x+4sa.
\]

The smallest determinant-safe amplitude is therefore

\[
a^*=\max\left(0,
\left\lceil\frac{m-sx}{4}\right\rceil
\right).
\]

### Principle

This is a closed-form minimum-amplitude solution to the sign-margin constraint. It explains the high PSNR without relying on post-hoc repair loops.

## Component 5: masked payload

### Object

A keyed binary mask \(k_i\) produces coded bits

\[
c_i=w_i\oplus k_i.
\]

### Principle

The host-dependent selected signs should not directly reveal the original watermark. The pilot authorizes removal of the mask during blind extraction.

## Component 6: determinant pilot

### Object

A separate set of determinant carriers is embedded with positive margin.

### Principle

The pilot and payload are expressed in the same statistic. Therefore, a geometric transform that restores pilot consistency should also restore the payload grid.

Reducing the pilot count from 79 to 71 lowers distortion while retaining sufficient spatial evidence.

## Component 7: affine synchronization as hypothesis testing

### Object

For candidate transform \(T\), let pilot differences be \(x_i(T)\). Define

\[
S(T)=\frac1{N_p}\sum_{i=1}^{N_p}
\tanh\left(
\frac{x_i(T)}{\operatorname{median}_j|x_j(T)|+\varepsilon}
\right).
\]

### Decision rule

Identity is the null hypothesis. A candidate is accepted only if

\[
S(T^*)-S(I)>0.30
\]

and

\[
S(T^*)\geq0.50.
\]

### Principle

The gain condition rejects unnecessary interpolation. The absolute-score condition prevents an unwatermarked image from being accepted merely because a transform improves a weak random pilot score.

## Measured evidence

Across 13 hosts:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Mean PSNR | 53.454653 dB | 53.636001 dB | +0.181348 dB |
| Moderate mean NC | 0.974359 | 0.974351 | -0.000008 |
| Rotation/shear mean NC | 0.546462 | 0.986670 | +0.440207 |
| Combined 17-attack mean NC | 0.924018 | 0.975800 | +0.051782 |

The moderate-set difference is negligible, while geometric robustness and PSNR improve. The change is accepted.

## Defensible novelty

> A channel-differential signed QR determinant with an exact antisymmetric minimum-amplitude update, coupled to a determinant-domain affine synchronization test.

The novelty is the relation among the carrier, update law, and synchronization statistic—not QR decomposition alone.

---

# 4. Scientific acceptance rules used by the repository

A modification is accepted only when it satisfies the relevant conditions.

## Clean correctness

\[
\operatorname{NC}_{\text{clean}}=1.
\]

## Joint quality-robustness improvement

At least one quality metric and one robustness metric must improve, and no important aggregate metric may deteriorate materially.

## Independent secondary-channel rule

A rescue channel must be tested before fusion:

\[
\operatorname{Accuracy}_{\text{secondary,clean}}\geq0.99.
\]

## Geometric synchronization rule

Transformation selection must use pilot evidence only, not the original watermark or original host.

## Reproducibility

The measured tables are stored in:

- `results/scientific_validation/FINAL_IMPROVEMENT_VALIDATION.md`
- `results/scientific_validation/final_validation.json`
- `results/scientific_validation/proposal_before_after_per_host.csv`

## Current non-DCT spatial QR extensions

The current registry adds three spatial QR counterparts that do not execute DCT/IDCT:

1. `spatial_qr`: \(v=(Y[1,2]-Y[2,1])/2\), QR reliability scheduling, pairwise coset optimization, and `R11` gain normalization.
2. `spatial_qr_direct_r`: direct QR of the central 4×4 spatial luminance patch and parity-QIM on \((R_{12}-R_{13})/2\). The antisymmetric update \((+\delta,-\delta)\) is the minimum-Frobenius two-entry update.
3. `spatial_qr_r11_qim`: direct parity-QIM on canonical positive \(R_{11}\) of the spatial QR factorization.

See `SPATIAL_QR_PROPOSAL_VI.md`, `SPATIAL_QR_DIRECT_R_PROPOSAL_VI.md`, `SPATIAL_QR_R11_QIM_PROPOSAL_VI.md`, and `SIX_QR_FAMILY_PROPOSALS.md` for the complete definitions.
