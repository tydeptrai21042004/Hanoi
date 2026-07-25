# DCT–QR Carrier-Subspace Gain-Normalized QIM

## 1. Research objective

The objective is to increase watermarked-image PSNR without reducing clean extraction accuracy or aggregate attacked normalized correlation (NC). The method remains strictly in the required **DCT + QR** domain:

- the watermark is embedded in a DCT differential carrier;
- QR reliability allocates local QIM spacing;
- a carrier-specific QR factor estimates local attack gain;
- extraction does not require the original host image.

The validated proposal is named **DCT–QR Carrier-Subspace Gain-Normalized QIM**, abbreviated **QR-CSGN-QIM**.

## 2. Research gap

### 2.1. Gap in conventional QIM adaptation

QIM provides a principled rate–distortion–robustness framework, while distortion-compensated and perceptually adaptive variants tune embedding strength using channel or perceptual models. However, a generic block reliability value does not necessarily estimate the transformation experienced by the particular coefficient combination that carries a bit.

### 2.2. Gap in QR watermarking

Representative QR watermarking methods commonly embed by quantizing or modifying selected entries or relations in the matrices `Q` or `R`. Such methods make QR the embedding domain itself. In contrast, the present method keeps the payload in a DCT-QIM carrier and uses QR as a **local transfer estimator**.

### 2.3. Gap in the previous implementation

The previous decoder used

\[
 s_b^{\mathrm{global}}=\|\operatorname{diag}(R_b)\|_2.
\]

This quantity averages all QR directions of a lifted analysis matrix. Only one low-frequency analysis column contains the DCT payload carrier. Energy changes in unrelated columns therefore contaminate the estimated gain. To preserve robustness, the previous method required relatively large QIM steps:

\[
(\Delta_{\mathrm{weak}},\Delta_{\mathrm{mid}},\Delta_{\mathrm{strong}})
=(20,15,12).
\]

The missing component was a QR statistic aligned with the actual carrier subspace.

## 3. DCT embedding model

For every \(8\times8\) image block \(b\), let \(C_b(u,v)\) be the orthonormal 2-D DCT coefficients of the opponent-color analysis field. The payload carrier is

\[
 v_b=\frac{1}{2}\bigl(C_b(0,1)-C_b(1,0)\bigr).
\]

For watermark bit \(m_b\in\{0,1\}\), local step \(\Delta_b>0\), and safety margin \(0\le \rho_b<\Delta_b/2\), define the safe parity set

\[
\mathcal S_{m_b}(\Delta_b,\rho_b)
=
\bigcup_{k\equiv m_b\ (\mathrm{mod}\ 2)}
\left[
 k\Delta_b-\frac{\Delta_b}{2}+\rho_b,
 k\Delta_b+\frac{\Delta_b}{2}-\rho_b
\right].
\]

The exact embedding law is the nearest Euclidean projection

\[
\boxed{
 v_b^*=\Pi_{\mathcal S_{m_b}(\Delta_b,\rho_b)}(v_b)
}
\]

followed by the minimum-norm RGB update and an integer-lattice closure. The closure makes the uint8 watermarked image itself satisfy the parity decision, explaining why clean NC is exactly one rather than only approximately one.

## 4. QR reliability allocation

A low-frequency DCT analysis matrix \(A_b\in\mathbb R^{4\times4}\) is formed and factorized using a canonical QR convention:

\[
 A_b=Q_bR_b,
\]

where the diagonal of \(R_b\) is made nonnegative. Reliability combines a determinant-based lower-bound term, diagonal balance, and upper-triangular coupling:

\[
\beta_b=\frac{|\det A_b|}{\|A_b\|_F^3+\varepsilon},
\]

\[
b_b=\frac{\min_i |r_{ii,b}|}{\max_i |r_{ii,b}|+\varepsilon},
\qquad
c_b=\frac{\|\operatorname{triu}(R_b,1)\|_F}{\|R_b\|_F+\varepsilon}.
\]

The normalized score ranks blocks into weak, middle, and strong groups. The new validated step levels are

\[
\boxed{
(\Delta_{\mathrm{weak}},\Delta_{\mathrm{mid}},\Delta_{\mathrm{strong}})
=(18.25,13.25,10.25)
}
\]

with proportions \(20\%/40\%/40\%\). These are smaller than the previous levels because the new decoder estimates carrier attenuation more accurately.

## 5. Carrier-specific QR transfer estimator

### 5.1. Definition

Construct the **unlifted** analysis matrix and order its first column so that it contains the low-frequency DCT direction associated with the payload carrier. For the canonical QR factorization,

\[
 A_b=Q_bR_b,
\]

the first diagonal element satisfies exactly

\[
\boxed{r_{11,b}=\|A_b[:,1]\|_2}
\]

using one-based mathematical indexing. The carrier-specific reference scale is

\[
 s_{b,0}^{\mathrm{car}}=|r_{11,b}^{(w)}|,
\]

computed from the final watermarked image and stored in the secret extraction key. No coefficient from the original host image is required.

For a questioned image,

\[
 s_b^{\mathrm{car}}=|r_{11,b}^{(q)}|,
\qquad
\widehat\alpha_b=
\operatorname{clip}
\left(
\frac{s_b^{\mathrm{car}}}{s_{b,0}^{\mathrm{car}}},
0.55,1.45
\right).
\]

The corrected carrier is

\[
\boxed{
\widetilde v_b=
\frac{v_b^{(q)}}{\widehat\alpha_b^{\gamma}},
\qquad \gamma=0.90.
}
\]

The extracted bit is

\[
\widehat m_b=
\operatorname{round}
\left(
\frac{\widetilde v_b}{\Delta_b}
\right)\bmod2.
\]

### 5.2. Exact multiplicative property

If the carrier-bearing column is scaled without additive error,

\[
 a'_{1,b}=\alpha_ba_{1,b},
\qquad \alpha_b>0,
\]

then

\[
 r'_{11,b}=\|a'_{1,b}\|_2
=\alpha_b\|a_{1,b}\|_2
=\alpha_br_{11,b}.
\]

Therefore,

\[
\frac{r'_{11,b}}{r_{11,b}}=\alpha_b.
\]

This is a carrier-aligned transfer identity; the old global diagonal norm does not isolate this direction.

### 5.3. Perturbation bound

Let

\[
 a'_{1,b}=\alpha_ba_{1,b}+e_b.
\]

By the reverse triangle inequality,

\[
\bigl|\|\alpha_ba_{1,b}+e_b\|_2
-\alpha_b\|a_{1,b}\|_2\bigr|
\le \|e_b\|_2.
\]

Dividing by \(r_{11,b}=\|a_{1,b}\|_2>0\) gives

\[
\boxed{
\left|
\frac{r'_{11,b}}{r_{11,b}}-\alpha_b
\right|
\le
\frac{\|e_b\|_2}{r_{11,b}}.
}
\]

Hence large carrier-column energy improves gain-estimation stability, and the error has a direct blockwise bound.

### 5.4. Why the exponent is regularized

Assume an attacked carrier

\[
 y_b=\alpha_bv_b+n_b
\]

and a multiplicative estimate

\[
\widehat\alpha_b=\alpha_be^{\epsilon_b}.
\]

Then

\[
\frac{y_b}{\widehat\alpha_b^\gamma}
=
\alpha_b^{1-\gamma}e^{-\gamma\epsilon_b}v_b
+
\frac{n_b}{\widehat\alpha_b^\gamma}.
\]

Full compensation \(\gamma=1\) removes ideal gain but maximizes sensitivity to estimation error and may amplify noise. For a simple independent log-gain model with variances \(\sigma_g^2\) and \(\sigma_\epsilon^2\), minimizing the approximate residual log-gain variance gives

\[
\boxed{
\gamma^*=\frac{\sigma_g^2}{\sigma_g^2+\sigma_\epsilon^2}.
}
\]

Thus \(0<\gamma<1\) is a shrinkage estimator rather than an arbitrary engineering constant. The validated finite grid selected \(\gamma=0.90\).

## 6. Minimum-norm color-domain update

Let the opponent-field color gradient be

\[
 g_\eta=
\begin{bmatrix}
0.299+\eta\\
0.587-\eta/2\\
0.114-\eta/2
\end{bmatrix}
\]

and let \(h=B_{01}-B_{10}\) denote the spatial DCT basis difference. For desired carrier displacement \(d_b=v_b^*-v_b\), the continuous update

\[
\boxed{
\Delta X_b
=d_b\,h\otimes\frac{g_\eta}{\|g_\eta\|_2^2}
}
\]

is the minimum-Frobenius-norm RGB perturbation satisfying the required linear carrier change. This follows from orthogonal projection onto the linear constraint normal, or equivalently from Cauchy–Schwarz. Integer closure then selects a nearby uint8 solution that exactly preserves the clean QIM parity.

## 7. Validated results

Protocol:

- 13 RGB host images of size \(512\times512\);
- one fixed \(64\times64\) binary watermark;
- fixed seed 2026;
- 15 deterministic moderate attacks;
- identical attack and host sets for the previous and proposed configurations.

| Metric | Previous global-QR decoder | Proposed carrier-QR decoder | Change |
|---|---:|---:|---:|
| Mean PSNR | 46.213584 dB | **47.250421 dB** | **+1.036837 dB** |
| Clean NC | 1.000000 | **1.000000** | 0 |
| Mean attacked NC | 0.997246 | **0.997867** | +0.000621 |
| Mean host-wise worst NC | 0.982613 | **0.986162** | +0.003549 |
| Global worst NC | 0.948264 | **0.964654** | +0.016390 |

The proposal improves mean NC for 14 of the 15 original attacks. JPEG Q70 has a negligible mean reduction of approximately \(7.5\times10^{-5}\), while clean NC remains exactly one and aggregate attacked NC increases.

A broader 60-attack diagnostic on three difficult hosts also improved aggregate behavior:

| Metric | Previous | Proposed |
|---|---:|---:|
| Mean PSNR | 46.379344 dB | **47.370438 dB** |
| Clean NC | 1.000000 | **1.000000** |
| Mean attacked NC | 0.925739 | **0.935093** |
| Lower-decile NC | 0.635072 | **0.702804** |
| Mean worst NC | 0.462288 | **0.537837** |
| Global worst NC | 0.410969 | **0.465765** |

Strong contrast amplification and some high-variance noise cases remain limitations. Consequently, the defensible claim is an aggregate robustness improvement under the declared protocol, not uniform domination under every possible attack.

## 8. Scoped novelty claim

A defensible paper claim is:

> We propose a carrier-subspace QR transfer estimator for DCT-QIM watermarking. Unlike QR watermarking methods that directly quantize selected entries of Q or R, and unlike global block-reliability adaptation, the proposed canonical \(r_{11}\) ratio estimates attenuation specifically along the DCT analysis column carrying the payload. A perturbation bound quantifies the estimator error, while regularized gain compensation creates sufficient robustness margin to reduce the local QIM steps and improve PSNR without reducing clean NC or aggregate attacked NC.

Do not claim that QIM, adaptive QIM, QR watermarking, or QR perturbation theory is itself new. The novelty is the **carrier-aligned coupling** of these components.

## 9. Required publication-final extensions

The current result is a strong research prototype, but publication-final validation should add:

1. at least three independent watermark patterns and three seeds;
2. held-out parameter selection separate from the final host set;
3. paired confidence intervals and a significance test for PSNR/NC changes;
4. key-size and runtime analysis;
5. ablations for global `diag_l2`, carrier `r11`, no compensation, and several \(\gamma\) values;
6. a strong-noise/contrast robustness section that reports the observed trade-offs;
7. comparisons against paper-faithful QIM and QR-domain watermarking baselines.

## 10. Theoretical references

1. B. Chen and G. W. Wornell, “Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding,” *IEEE Transactions on Information Theory*, 47(4), 2001, DOI: 10.1109/18.923725.
2. J. J. Eggers, R. Bäuml, R. Tzschoppe, and B. Girod, “Scalar Costa Scheme for Information Embedding,” *IEEE Transactions on Signal Processing*, 51(4), 2003, DOI: 10.1109/TSP.2003.809384.
3. Q. Su, Y. Niu, H. Zou, Y. Zhao, and T. Yao, “Color Image Blind Watermarking Scheme Based on QR Decomposition,” *Signal Processing*, 94, 219–235, 2014, DOI: 10.1016/j.sigpro.2013.06.025.
4. G. W. Stewart, “Perturbation Bounds for the QR Factorization of a Matrix,” *SIAM Journal on Numerical Analysis*, 14(3), 509–518, 1977, DOI: 10.1137/0714030.
5. X.-W. Chang, C. C. Paige, and G. W. Stewart, “Perturbation Analyses for the QR Factorization,” *SIAM Journal on Matrix Analysis and Applications*, 18(3), 775–791, 1997, DOI: 10.1137/S0895479896297720.
