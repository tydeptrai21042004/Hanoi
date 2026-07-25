# Ba proposal watermarking được tái thiết kế

## 1. Mục tiêu và ràng buộc miền

Bản tái thiết kế giữ nguyên ba miền khoa học bắt buộc:

1. **DCT–QR Reliability-Conditioned QIM**: mọi bit được nhúng bằng DCT-QIM; QR phải tham gia trực tiếp vào phân bổ cường độ và trích xuất.
2. **DCT–Schur Spectral-Gain QIM**: mọi bit được nhúng bằng DCT-QIM; Schur phải tham gia trực tiếp vào phân bổ cường độ và bù suy hao, không chỉ làm một kênh phụ yếu.
3. **Spatial Normalized-Residual DetQR**: phương pháp hoàn toàn trong miền không gian; QR và điều kiện cứng \(\det A\ne0\) phải nằm ngay trong luật nhúng.

Các mục tiêu kiểm chứng là:

- NC sạch bằng đúng 1;
- PSNR trung bình cao hơn phiên bản proposal cũ;
- NC trung bình trên tập tấn công cao hơn phiên bản cũ;
- có luật nhúng rõ ràng, có mệnh đề/toán học giải thích thành phần mới;
- code, cấu hình và script đánh giá thống nhất.

> **Phạm vi tuyên bố:** “mới” trong tài liệu này là điểm mới của luật ghép/cơ chế đề xuất so với implementation cũ và các họ phương pháp liên quan đã khảo sát. Không tuyên bố “đầu tiên trên thế giới” nếu chưa có systematic review và tra cứu bằng sáng chế hoàn chỉnh.

---

## 2. Giao thức kiểm chứng

- 13 ảnh host RGB, kích thước \(512\times512\).
- Watermark nhị phân \(64\times64\), tương đương 4096 bit.
- 15 tấn công mức vừa, cùng attack suite cho cả ba phương pháp.
- 39 lần clean round-trip: 13 host × 3 proposal.
- Cấu hình được lấy từ các file `configs/*_after_pso.json`.
- Unit/integration tests: **29 passed**.

Kết quả tổng hợp:

| Proposal mới | PSNR trung bình (dB) | Clean NC | Mean NC, 15 attacks | Mean worst NC/host | Global worst NC |
|---|---:|---:|---:|---:|---:|
| DCT–QR | 46.213584 | 1.000000 | 0.997246 | 0.982613 | 0.948264 |
| DCT–Schur | 48.014086 | 1.000000 | 0.996058 | 0.975353 | 0.923018 |
| Spatial DetQR | 56.670744 | 1.000000 | 0.991345 | 0.942829 | 0.915258 |

So với số liệu proposal cũ đã lưu trong repository:

| Proposal | \(\Delta\)PSNR | \(\Delta\)Mean NC | \(\Delta\)Mean worst NC |
|---|---:|---:|---:|
| DCT–QR | +0.528985 dB | +0.006256 | +0.019648 |
| DCT–Schur | +0.982801 dB | +0.007256 | +0.017814 |
| Spatial DetQR | +3.034743 dB | +0.016994 | +0.186238 |

Lưu ý trung thực: Spatial DetQR tăng mạnh PSNR và NC tổng hợp, nhưng kiểm tra riêng hai phép biến đổi hình học lớn giảm nhẹ so với phiên bản cũ:

- rotation \(2^\circ\): 0.992574 → 0.985366;
- shear 0.08: 0.980765 → 0.976092.

Do đó không được viết rằng **mọi tấn công riêng lẻ** đều tốt hơn; tuyên bố đúng là chất lượng và độ bền **tổng hợp trên giao thức 15 tấn công** tăng.

---

# 3. Proposal I — DCT–QR Gain-Normalized Reliability QIM

## 3.1. Miền và carrier DCT

Từ ảnh RGB, tạo trường đối nghịch:

\[
F=0.299R+0.587G+0.114B
+\eta\left(R-\frac{G+B}{2}\right),\qquad \eta=0.07.
\]

Với block \(8\times8\), thực hiện DCT trực chuẩn và định nghĩa carrier:

\[
v_b=\frac{1}{2}\bigl(C_b(0,1)-C_b(1,0)\bigr).
\]

Bit \(m_b\in\{0,1\}\) được mã hóa bằng hai coset QIM chẵn/lẻ. Với bước cục bộ \(\Delta_b\), chỉ số lattice mục tiêu có parity bằng \(m_b\):

\[
k_b^*=\arg\min_{k\in\mathbb Z,\;k\bmod 2=m_b}
|v_b-k\Delta_b|,
\qquad
v_b^*=k_b^*\Delta_b.
\]

Implementation tiếp tục dùng integer-lattice closure để bảo đảm ảnh `uint8` cuối cùng vẫn giải mã sạch chính xác.

## 3.2. Ma trận phân tích QR

Từ 16 hệ số DCT tần số thấp của block, xây dựng ma trận nâng \(A_b\in\mathbb R^{4\times4}\). Hệ số DC được thay bằng hiệu định hướng \(C_b(0,1)-C_b(1,0)\), sau đó thêm diagonal lift dương để ổn định phân tích:

\[
A_b=Q_bR_b,
\]

với QR canonical, tức đường chéo của \(R_b\) không âm.

Reliability được tạo từ ba thành phần:

\[
\beta_b=\frac{|\det A_b|}{\|A_b\|_F^3},
\]

\[
b_b=\frac{\min_i |r_{ii}|}{\max_i |r_{ii}|+\varepsilon},
\qquad
c_b=\frac{\|\operatorname{triu}(R_b,1)\|_F}
{\|R_b\|_F+\varepsilon},
\]

\[
r_b=\operatorname{Norm}_{5\%,95\%}
\left[\log(1+\beta_b)+2b_b+0.2(1-c_b)\right].
\]

## 3.3. Luật phân bổ QIM theo QR

QR reliability chia block thành ba nhóm. Cấu hình đã kiểm chứng sử dụng:

\[
\Delta_b=
\begin{cases}
20,&r_b\text{ thuộc 20\% yếu nhất},\\
15,&r_b\text{ thuộc 40\% giữa},\\
12,&r_b\text{ thuộc 40\% mạnh nhất}.
\end{cases}
\]

Ý nghĩa: block yếu nhận khoảng cách coset lớn hơn để giảm lỗi; block ổn định nhận bước nhỏ hơn để giảm méo. Đây là một phân bổ méo rời rạc theo reliability thay vì một \(\Delta\) đồng nhất.

## 3.4. Đóng góp QR mới: bù gain từ canonical \(R\)

Định nghĩa thang QR:

\[
s_b^{QR}=\|\operatorname{diag}(R_b)\|_2.
\]

Tham chiếu \(s_b^{QR,0}\) được tính từ **ảnh watermarked cuối cùng**, không lấy từ ảnh host gốc. Khi trích xuất ảnh nghi vấn, tính:

\[
\alpha_b^{QR}
=\operatorname{clip}\left(
\frac{s_b^{QR}}{s_b^{QR,0}},0.55,1.45
\right),
\]

\[
\widetilde v_b=rac{v_b}{(\alpha_b^{QR})^{\gamma}},
\qquad \gamma=0.75.
\]

Sau đó giải mã parity từ \(\widetilde v_b/\Delta_b\).

### Mệnh đề 1 — Tính đồng bậc của thang QR

Nếu block sau biến đổi gain lý tưởng thỏa:

\[
A_b'=\alpha A_b,\qquad \alpha>0,
\]

thì với QR canonical:

\[
A_b'=Q_b(\alpha R_b),
\]

và do đó:

\[
s_b^{QR}(A_b')=\alpha s_b^{QR}(A_b).
\]

Carrier DCT tuyến tính cũng thỏa \(v_b'=\alpha v_b\). Vì vậy với \(\gamma=1\), phép chia bởi tỷ số QR khử chính xác gain lý tưởng. Giá trị \(\gamma=0.75\) là shrinkage thực nghiệm để tránh over-correction khi attack không phải gain thuần.

## 3.5. Điểm mới có thể bảo vệ

Không tuyên bố mới ở việc “dùng QR với DCT”, vì đã có nhiều watermarking scheme dùng QR. Điểm mới cụ thể là:

1. **Reliability-conditioned QIM spacing** từ determinant bound, diagonal balance và QR coupling.
2. **Canonical-R gain normalization** dùng tỷ số thang QR của ảnh watermarked và ảnh nghi vấn.
3. Ghép hai vai trò QR trong cùng một pipeline: QR điều khiển cả **méo khi nhúng** và **bù suy hao khi trích xuất**.

## 3.6. Kết quả

- PSNR: 45.684599 → **46.213584 dB**.
- Mean NC: 0.990990 → **0.997246**.
- Mean worst NC: 0.962965 → **0.982613**.
- Clean NC: **1.000000** trên 13/13 host.

---

# 4. Proposal II — DCT–Schur Spectral-Gain QIM

## 4.1. Vì sao bỏ kênh Schur phụ cũ

Phiên bản cũ nhúng một kênh bit phụ bằng departure from normality rồi fusion với DCT. Kênh Schur độc lập không đạt tiêu chuẩn clean accuracy 0.99; vì vậy nó không đủ mạnh để làm contribution chính.

Bản mới giữ code cũ chỉ để ablation/negative result. Public proposal **không nhúng và không fusion kênh phụ đó**. Thay vào đó, Schur tham gia trực tiếp vào mọi bit DCT-QIM thông qua:

1. phân bổ \(\Delta_b\) theo Schur reliability;
2. ước lượng gain cục bộ từ các đại lượng Schur đồng bậc.

## 4.2. Phân tích Schur

Với ma trận DCT nâng \(A_b\), dùng complex Schur:

\[
A_b=Z_bT_bZ_b^H,
\]

trong đó đường chéo \(T_b\) chứa các trị riêng \(\lambda_i(A_b)\).

Henrici departure from normality được tính bởi:

\[
\operatorname{dep}(A_b)^2
=\|A_b\|_F^2-\sum_i|\lambda_i(A_b)|^2.
\]

Trong complex Schur, đại lượng này tương ứng năng lượng strict-upper của \(T_b\).

Schur reliability dùng:

\[
b_b^S=\frac{\min_i|\lambda_i|}{\max_i|\lambda_i|+\varepsilon},
\]

\[
c_b^S=\frac{\operatorname{dep}(A_b)}{\|A_b\|_F+\varepsilon},
\]

kết hợp với \(\beta_b\) để phân nhóm độ ổn định.

## 4.3. Luật QIM theo Schur

Bước cục bộ đã kiểm chứng:

\[
\Delta_b=
\begin{cases}
16.5,&20\%\text{ block Schur yếu nhất},\\
12.5,&40\%\text{ block giữa},\\
9.0,&40\%\text{ block mạnh nhất}.
\end{cases}
\]

Luật nhúng DCT vẫn là nearest parity-lattice như Proposal I, nhưng mọi \(\Delta_b\) được quyết định bởi Schur certificate.

## 4.4. Thang gain Schur phổ–departure

Định nghĩa:

\[
s_b^S=
\sqrt{
\sum_i|\lambda_i(A_b)|^2
+\omega\operatorname{dep}(A_b)^2
},
\qquad \omega=0.5.
\]

Khi trích xuất:

\[
\alpha_b^S=
\operatorname{clip}\left(
\frac{s_b^S}{s_b^{S,0}},0.55,1.45
\right),
\]

\[
\widetilde v_b=rac{v_b}{(\alpha_b^S)^{0.75}}.
\]

### Mệnh đề 2 — Tính đồng bậc của thang Schur

Với mọi \(\alpha\in\mathbb C\):

\[
\lambda_i(\alpha A)=\alpha\lambda_i(A),
\]

và:

\[
\operatorname{dep}(\alpha A)=|\alpha|\operatorname{dep}(A).
\]

Suy ra:

\[
s^S(\alpha A)=|\alpha|s^S(A).
\]

Do đó tỷ số \(s_b^S/s_b^{S,0}\) là estimator tự nhiên cho suy hao cục bộ trong mô hình gain. Thành phần departure giữ thông tin về non-normal coupling, thay vì chỉ dùng năng lượng trị riêng.

## 4.5. Điểm mới có thể bảo vệ

Không tuyên bố “DCT + Schur đầu tiên”; các nghiên cứu trước đã kết hợp DCT và Schur để nhúng watermark. Điểm mới cụ thể là:

1. **Schur spectral reliability-conditioned QIM allocation**.
2. **Homogeneous spectral–departure gain estimator** điều khiển trực tiếp biến DCT-QIM khi giải mã.
3. Thay một kênh Schur bit phụ không ổn định bằng Schur side information có cơ sở đồng bậc và áp dụng cho toàn bộ payload.

## 4.6. Kết quả

- PSNR: 47.031285 → **48.014086 dB**.
- Mean NC: 0.988802 → **0.996058**.
- Mean worst NC: 0.957539 → **0.975353**.
- Clean NC: **1.000000** trên 13/13 host.

---

# 5. Proposal III — Spatial Normalized-Residual DetQR

## 5.1. Ma trận QR không gian và điều kiện \(\det A\ne0\)

Mỗi block \(8\times8\) được xét với tối đa 10 phân hoạch Hadamard có boundary thấp. Với một pattern, ký hiệu:

\[
u=\operatorname{mean}(Y\mid P=+1),
\qquad
v=\operatorname{mean}(Y\mid P=-1),
\]

và xây dựng:

\[
A=\begin{bmatrix}u&1\\v&1\end{bmatrix}=QR.
\]

Ta có:

\[
\det A=u-v.
\]

Nếu chỉ dùng \(\det A\), QR là dư thừa đại số. Bản mới dùng đại lượng QR chuẩn hóa:

\[
z(A)=\det(Q)r_{22}
=\frac{\det A}{r_{11}}
=\frac{u-v}{\sqrt{u^2+v^2}}.
\]

Vì \(r_{11}=\sqrt{u^2+v^2}\), QR đóng vai trò thực sự: carrier không chỉ là hiệu mean, mà là residual determinant đã chuẩn hóa bởi năng lượng cột đầu của \(A\).

## 5.2. Luật cập nhật không gian

Với bit đã mask, đặt dấu mong muốn \(s\in\{-1,+1\}\). Áp dụng cập nhật nguyên phản đối xứng:

\[
u'=u+sa,\qquad v'=v-sa,
\qquad a\in\mathbb Z_{\ge0}.
\]

Đặt:

\[
d=u-v,\qquad k=u+v.
\]

Khi đó:

\[
d'=d+2sa,
\qquad k'=k.
\]

Mục tiêu nhúng gồm hai ràng buộc cứng:

\[
sz(A')\ge\mu,
\]

\[
|\det A'|\ge\delta>0.
\]

## 5.3. Định lý biên độ nguyên tối thiểu

Vì:

\[
z(A')=
\frac{\sqrt2\,d'}{\sqrt{k^2+d'^2}},
\]

điều kiện \(sz(A')\ge\mu\), với \(0<\mu<\sqrt2\), tương đương:

\[
sd'\ge
\frac{\mu|k|}{\sqrt{2-\mu^2}}.
\]

Kết hợp với determinant floor, đặt:

\[
T=\max\left(
\delta,
\frac{\mu|k|}{\sqrt{2-\mu^2}}
\right).
\]

Biên độ nguyên không âm nhỏ nhất là:

\[
\boxed{
 a^*=\max\left(
0,
\left\lceil\frac{T-sd}{2}\right\rceil
\right)
}.
\]

### Mệnh đề 3 — Tính khả thi và tối thiểu

- Với \(a=a^*\), ta có \(sd'=sd+2a^*\ge T\), do đó đồng thời thỏa normalized QR margin và \(|\det A'|\ge\delta\).
- Nếu \(a^*>0\), mọi số nguyên \(a<a^*\) thỏa \(sd+2a<T\), nên vi phạm ít nhất một trong hai ràng buộc.

Vì vậy luật trên là nghiệm tối thiểu chính xác cho biên độ nguyên theo mô hình block mean.

## 5.4. Chọn pattern và pilot

Với mỗi payload block, tính \(a^*\) cho từng Hadamard pattern rồi chọn:

\[
p^*=\arg\min_p
\left[(a_p^*)^2+\lambda B_p-10^{-6}s z_p\right],
\]

trong đó \(B_p\) là boundary complexity. Term chính \((a_p^*)^2\) ưu tiên distortion thấp nhất; boundary penalty tránh pattern dao động quá mạnh.

Pilot dùng cùng miền normalized QR residual, cùng điều kiện \(\det A\ne0\), nhưng ở các pattern khác payload. Affine correction chỉ được chấp nhận khi pilot score tăng đủ lớn và vượt ngưỡng tuyệt đối.

## 5.5. Điểm mới có thể bảo vệ

1. **QR-active carrier** \(z(A)=\det(Q)r_{22}=\det(A)/r_{11}\), khắc phục việc QR chỉ là cách tính lại determinant.
2. **Closed-form minimum integer embedding law** thỏa đồng thời signed QR margin và hard nonsingularity.
3. **Unified payload–pilot QR domain**: payload và affine synchronization dùng cùng thống kê QR có determinant floor.

## 5.6. Kết quả

- PSNR: 53.636001 → **56.670744 dB**.
- Mean NC: 0.974351 → **0.991345**.
- Mean worst NC: 0.756591 → **0.942829**.
- Clean NC: **1.000000** trên 13/13 host.
- Rotation \(2^\circ\): 0.985366.
- Shear 0.08: 0.976092.

---

# 6. Ranh giới với công trình liên quan

Các họ công trình nền tảng cần được trích dẫn trong bài báo:

1. **QIM:** B. Chen và G. W. Wornell, “Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding,” *IEEE Transactions on Information Theory*, 2001.
2. **Blind QR watermarking:** Q. Su và cộng sự, “Color image blind watermarking scheme based on QR decomposition,” *Signal Processing*, 2014.
3. **DCT–Schur watermarking:** A. Soualmi và cộng sự, “Schur and DCT Decomposition Based Medical Images Watermarking,” 2018; cùng các nghiên cứu Schur watermarking trước/sau đó.
4. **Departure from normality:** các kết quả Henrici về departure from normality và liên hệ với năng lượng strict-upper của Schur form.

Do prior art đã có DCT, QR và Schur watermarking, title/abstract nên tập trung vào luật mới:

- “QR gain-normalized reliability-conditioned QIM”;
- “Schur spectral-departure gain-compensated QIM”;
- “minimum-integer normalized-residual DetQR embedding under a nonsingularity constraint”.

---

# 7. Những gì đã được kiểm chứng và chưa được kiểm chứng

## Đã kiểm chứng

- Code chạy và 29 test pass.
- Clean NC bằng 1 trên 39 clean runs.
- PSNR và aggregate mean NC tăng cho cả ba proposal so với reference cũ.
- Spatial theorem được unit-test: \(a^*\) thỏa điều kiện và \(a^*-1\) thất bại khi \(a^*>0\).
- QR residual trong code khớp \(\det(A)/r_{11}\).
- Public DCT–Schur path không còn dùng legacy Schur secondary vote.

## Chưa đủ để tuyên bố publication-final

- Chỉ mới dùng một watermark pattern và một bộ seed trong validation chính.
- Tham số chưa được tách rõ train/validation/test host.
- Chưa có confidence interval và paired significance test qua nhiều watermark/seed.
- Chưa benchmark đầy đủ với các phương pháp độc lập mạnh nhất trong literature dưới cùng protocol.
- Cả ba proposal hiện cố định cho host \(512\times512\), watermark \(64\times64\).
- Đây là **key-assisted blind extraction**: không dùng host gốc, nhưng key lưu lịch block và tham chiếu gain theo block; không nên gọi là tiny-key blind.

Để đưa vào paper, vòng thí nghiệm tiếp theo nên gồm ít nhất:

- 3–5 watermark khác nhau;
- 3–5 seed;
- tập host riêng cho chọn tham số và đánh giá cuối;
- ablation từng thành phần;
- bootstrap confidence interval và paired Wilcoxon/permutation test;
- runtime, key size và memory cost;
- các mức JPEG, blur, noise và geometric grid rộng hơn.

---

# 8. Tái lập

Cài đặt:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JILP_NUM_THREADS=1
```

Kiểm thử:

```bash
pytest -q
```

Chạy nhanh một host cho cả ba proposal:

```bash
python scripts/run_redesigned_validation.py --host-limit 1
```

Chạy đầy đủ 13 host:

```bash
python scripts/run_redesigned_validation.py
```

Kết quả xác nhận đi kèm:

- `results/redesigned_13host_validation.json`;
- `results/redesigned_spatial_geometry_13host.json`;
- `results/redesigned_comparison.json`.
