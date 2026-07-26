# Ba proposal watermarking được tái thiết kế

## 1. Mục tiêu và ràng buộc miền

Bản tái thiết kế giữ nguyên ba miền khoa học bắt buộc:

1. **DCT–QR Pairwise Coset-Optimized QIM**: QR điều khiển phân bổ bước, ghép cặp block tương đồng và chọn coset có méo nhỏ nhất; QR `r11` tiếp tục bù gain khi trích xuất.
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
| DCT–QR | **50.399139** | 1.000000 | **0.998028** | **0.988097** | **0.964723** |
| DCT–Schur | 48.014086 | 1.000000 | 0.996058 | 0.975353 | 0.923018 |
| Spatial DetQR | 56.670744 | 1.000000 | 0.991345 | 0.942829 | 0.915258 |

So với số liệu proposal cũ đã lưu trong repository:

| Proposal | \(\Delta\)PSNR | \(\Delta\)Mean NC | \(\Delta\)Mean worst NC |
|---|---:|---:|---:|
| DCT–QR | **+4.714540 dB** | **+0.007038** | **+0.025132** |
| DCT–Schur | +0.982801 dB | +0.007256 | +0.017814 |
| Spatial DetQR | +3.034743 dB | +0.016994 | +0.186238 |

Lưu ý trung thực: Spatial DetQR tăng mạnh PSNR và NC tổng hợp, nhưng kiểm tra riêng hai phép biến đổi hình học lớn giảm nhẹ so với phiên bản cũ:

- rotation \(2^\circ\): 0.992574 → 0.985366;
- shear 0.08: 0.980765 → 0.976092.

Do đó không được viết rằng **mọi tấn công riêng lẻ** đều tốt hơn; tuyên bố đúng là chất lượng và độ bền **tổng hợp trên giao thức 15 tấn công** tăng.

---

# 3. Proposal I — DCT–QR Pairwise Coset-Optimized QIM

## 3.1. Carrier và bước QIM

Từ trường đối nghịch RGB, với mỗi block \(8\times8\), carrier là

\[
v_b=\frac{C_b(0,1)-C_b(1,0)}{2}.
\]

QR reliability vẫn phân bổ ba bước đã kiểm chứng:

\[
\Delta_b\in\{18.25,13.25,10.25\},
\qquad \rho_b=0.45\Delta_b.
\]

## 3.2. Luật nhúng mới

Với bit watermark đã scramble \(u_b\), định nghĩa phép chiếu khoảng an toàn
\(P_c(v_b)\) lên coset QIM parity \(c\). Sắp xếp block theo QR reliability và
ghép cặp \(G_j\). Nhãn coset chung của cặp là

\[
\boxed{
s_j^*=\arg\min_{s\in\{0,1\}}
\sum_{b\in G_j}|P_{u_b\oplus s}(v_b)-v_b|^2.
}
\]

Bit thực sự nhúng và carrier mục tiêu là

\[
\boxed{c_b=u_b\oplus s_j^*,\qquad v_b^*=P_{c_b}(v_b).}
\]

Sau đó dùng cập nhật RGB chuẩn nhỏ nhất và integer-lattice closure hiện có.
Không giảm \(\Delta_b\) và không giảm \(\rho_b\).

## 3.3. Luật trích xuất

Sau bù gain bằng canonical \(r_{11}\),

\[
\widehat c_b=\operatorname{round}(\widetilde v_b/\Delta_b)\bmod2,
\qquad
\boxed{\widehat u_b=\widehat c_b\oplus s_j^*.}
\]

## 3.4. Cơ sở toán học

Vì mapping cũ tương ứng với lựa chọn khả thi \(s=0\), nghiệm tối ưu thỏa

\[
D_{\mathrm{new}}\le D_{\mathrm{old}}.
\]

Ngoài ra,

\[
\mathbf1[\widehat u_b\ne u_b]
=
\mathbf1[\widehat c_b\ne c_b],
\]

nên phép XOR hoàn nguyên không tạo thêm lỗi bit. Điều kiện đủ cũ
\(|e_b|<\rho_b\) vẫn giữ nguyên do bước và guard margin không thay đổi.

## 3.5. Kết quả

- PSNR: 47.250421 → **50.399139 dB** so với bản carrier-`r11` ngay trước đó.
- Clean NC: **1.000000**.
- Mean attacked NC: 0.997867 → **0.998028**.
- Mean worst NC/host: 0.986162 → **0.988097**.
- Global worst NC: 0.964654 → **0.964723**.

Key lưu flip mask đóng gói 4096 bit (512 byte thô), đại diện cho 2048 nhãn cặp.
Đây là side information cần báo cáo minh bạch trong bài báo.

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
