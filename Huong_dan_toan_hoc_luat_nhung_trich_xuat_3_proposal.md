# HƯỚNG DẪN TOÁN HỌC: LUẬT NHÚNG, LUẬT TRÍCH XUẤT VÀ CÁCH CHẠY 3 PROPOSAL

> **Repository được đối chiếu:** `Hanoi-main` trong file ZIP người dùng cung cấp.  
> **Mục tiêu tài liệu:** giải thích đúng implementation hiện tại, đặc biệt là *luật nhúng*, *luật trích xuất*, các công thức toán học và lệnh chạy tái lập.  
> **Ba proposal công khai:** `dct_qr`, `dct_schur_rescue`, `spatial_cd_detqr`.

---

## 1. Tổng quan nhanh

| ID | Tên phương pháp | Miền nhúng | Biến quyết định chính | Luật nhúng cốt lõi | Luật trích xuất cốt lõi |
|---|---|---|---|---|---|
| `dct_qr` | DCT–QR Pairwise Coset-Optimized Gain-Normalized QIM | DCT 8×8 | carrier vi sai \(v_b=(C_{01}-C_{10})/2\) | chiếu QIM theo parity, rồi chọn nhãn coset chung cho từng cặp để giảm distortion | hiệu chỉnh gain bằng QR-\(r_{11}\), làm tròn lattice, lấy parity, XOR ngược coset |
| `dct_schur_rescue` | DCT–Schur SP-SCQIM | DCT + tọa độ Schur strict-upper | 3 carrier trực giao \(h_k^Tn\) | \(n^*=n+H^T(t-Hn)\) | giải mã parity của 3 carrier, hiệu chỉnh gain, bỏ permutation, cộng phiếu và MAP |
| `spatial_cd_detqr` | Spatial Normalized-Residual DetQR | Không gian ảnh | dấu của \(\det(A)=u-v\) và QR residual chuẩn hóa | \(a^*=\max(0,\lceil(T-sd)/2\rceil)\) | đọc dấu determinant; nếu pilot hợp lệ thì bỏ mask payload |

Điểm quan trọng: **ba phương pháp này không phải ba biến thể của cùng một QIM**. Chúng có luật nhúng khác nhau về bản chất.

- `dct_qr`: bài toán **chiếu lên miền QIM parity**.
- `dct_schur_rescue`: bài toán **chiếu trực giao tối thiểu Frobenius lên ba ràng buộc tuyến tính**.
- `spatial_cd_detqr`: bài toán **tìm biên độ nguyên nhỏ nhất để ép dấu và biên QR/determinant**.

---

## 2. Ký hiệu chung

- Ảnh host RGB: \(I\in\{0,\ldots,255\}^{512\times512\times3}\).
- Watermark nhị phân: \(W\in\{0,1\}^{64\times64}\), tổng cộng \(4096\) bit.
- Mỗi ảnh 512×512 được chia thành \(64\times64=4096\) block 8×8, nên có thể gắn một bit payload cho mỗi block.
- \(b\): chỉ số block.
- \(C_b(i,j)\): hệ số DCT trực chuẩn tại vị trí \((i,j)\) của block \(b\).
- \(\Delta\): bước lượng tử/QIM step.
- \(\rho\): guard margin bên trong miền quyết định QIM.
- \(\widehat W\): watermark trích xuất.

Hai proposal DCT dùng trường màu

\[
F(R,G,B)=0.299R+0.587G+0.114B
+\eta\left(R-\frac{G+B}{2}\right),
\qquad \eta=0.07.
\]

Viết

\[
\mathbf g=
\begin{bmatrix}
0.299+\eta\\
0.587-\eta/2\\
0.114-\eta/2
\end{bmatrix}.
\]

Khi cần thay đổi trường \(F\) một lượng \(\delta\), hướng RGB có norm nhỏ nhất là

\[
\boxed{
\mathbf p=\frac{\mathbf g}{\|\mathbf g\|_2^2}
}
\]

vì \(\mathbf g^T\mathbf p=1\). Đây là lý do code có các hệ số `p0`, `p1`, `p2`.

---

# PHẦN I — DCT–QR PAIRWISE COSET-OPTIMIZED QIM

## 3. Ý tưởng của `dct_qr`

Phương pháp này giữ carrier DCT vi sai, nhưng bổ sung ba thành phần chính:

1. **QR reliability** để quyết định block nào cần QIM mạnh/yếu hơn.
2. **Pairwise coset optimization** để chọn cách gán parity ít gây méo nhất.
3. **QR carrier-\(r_{11}\) gain normalization** để bù suy hao cục bộ khi trích xuất.

Source chính:

```text
src/qr64_certified/proposals/method.py
src/qr64_certified/proposals/config.py
src/qr64_certified/proposals/certificate.py
src/qr64_certified/_jilp_core/engine.py
```

### 3.1 Carrier DCT

Với mỗi block 8×8, tính DCT trực chuẩn. Carrier payload là

\[
\boxed{
v_b=\frac{C_b(0,1)-C_b(1,0)}{2}.
}
\]

Đây là một low-frequency differential carrier: thay vì dùng trực tiếp một hệ số DCT, ta dùng hiệu của hai hệ số đối xứng thấp tần.

---

## 4. QR certificate và reliability

Từ block DCT, code tạo ma trận phân tích 4×4 \(A_b\). Thành phần DC được thay bởi hiệu \(C_{01}-C_{10}\), sau đó cộng một diagonal lift để phân tích độ ổn định.

Thực hiện QR chuẩn hóa dấu:

\[
A_b=Q_bR_b,
\]

với đường chéo \(R_b\) được ép không âm để loại bỏ bất định dấu của QR.

Các đại lượng độ ổn định:

\[
\beta_b=\frac{|\det(A_b)|}{\|A_b\|_F^3+\varepsilon},
\]

\[
\text{balance}_b=
\frac{\min_i |(R_b)_{ii}|}
{\max_i |(R_b)_{ii}|+\varepsilon},
\]

\[
\text{coupling}_b=
\frac{\|\operatorname{triu}(R_b,1)\|_F}
{\|R_b\|_F+\varepsilon}.
\]

Điểm reliability thô:

\[
r_b^{\rm raw}=\log(1+\beta_b)
+2\,\text{balance}_b
+0.2(1-\text{coupling}_b).
\]

Sau đó code chuẩn hóa theo quantile 5%–95% về \([0,1]\). Block có reliability thấp được xem là **kém ổn định**.

---

## 5. Luật chọn QIM step theo reliability

Payload block được sắp theo reliability từ thấp đến cao. Ba mức QIM được phân cho ba nhóm:

- nhóm yếu nhất: bước lớn nhất;
- nhóm trung bình: bước chuẩn;
- nhóm ổn định nhất: bước nhỏ nhất.

Ở **default public API** (`QR64Config.public_dct_qr()`):

\[
\boxed{
\Delta_b\in\{18.25,\ 13.25,\ 10.25\}
}
\]

với tỉ lệ nhóm xấp xỉ 20% / 40% / 40%, và

\[
\rho_b=0.45\Delta_b.
\]

**Lưu ý về config:** lệnh benchmark `--stage after_abc` không nhất thiết dùng đúng các giá trị default trên; nó đọc `configs/dct_qr_after_abc.json`. Trong ZIP hiện tại, ABC đã thay `step` cơ sở nhưng vẫn giữ cùng luật toán học và cùng `coset_group_size=2`.

---

## 6. Miền quyết định QIM có guard margin

Với bit mã hóa \(c\in\{0,1\}\), các lattice center hợp lệ là \(q\Delta_b\) với parity \(q\bmod2=c\).

Miền an toàn quanh center:

\[
\mathcal S_c(\Delta_b,\rho_b)=
\bigcup_{q\in\mathbb Z:\ q\bmod2=c}
\left[
q\Delta_b-\frac{\Delta_b}{2}+\rho_b,
q\Delta_b+\frac{\Delta_b}{2}-\rho_b
\right].
\]

Phép chiếu distortion nhỏ nhất:

\[
\boxed{
P_c(v_b)=
\arg\min_{x\in\mathcal S_c(\Delta_b,\rho_b)}
|x-v_b|^2.
}
\]

Khác center-QIM truyền thống, nếu \(v_b\) đã nằm trong vùng an toàn đúng parity thì **không cần di chuyển carrier**.

---

## 7. Luật nhúng mới: Pairwise Coset Optimization

Trước tiên watermark được Arnold-scramble thành chuỗi bit \(u_b\).

Các block payload được sắp theo QR reliability rồi chia thành từng cặp

\[
G_j=\{b_{j,1},b_{j,2}\}.
\]

Thay vì bắt buộc nhúng trực tiếp parity \(u_b\), mỗi cặp được phép dùng một nhãn chung \(s_j\in\{0,1\}\).

Luật chọn nhãn:

\[
\boxed{
s_j^*=\arg\min_{s\in\{0,1\}}
\sum_{b\in G_j}
\left|P_{u_b\oplus s}(v_b)-v_b\right|^2.
}
\]

Bit vật lý thực sự được nhúng:

\[
\boxed{
c_b=u_b\oplus s_j^*.
}
\]

Carrier sau nhúng:

\[
\boxed{
v_b^*=P_{c_b}(v_b).
}
\]

### Vì sao coset optimization giúp PSNR?

Cách cũ tương đương luôn chọn \(s=0\). Trong luật mới, \(s=0\) vẫn là một nghiệm khả thi, nên

\[
D_j^{\rm new}
=\min(D_j(s=0),D_j(s=1))
\le D_j(s=0)=D_j^{\rm old}.
\]

Do đó, ở mức **projection energy lý tưởng trước khi lượng tử RGB**, luật pairwise coset không thể làm distortion lớn hơn mapping cũ.

---

## 8. Từ thay đổi carrier sang thay đổi pixel RGB

Gọi

\[
\delta_b=v_b^*-v_b.
\]

Hai basis DCT payload là \(B_{01}\) và \(B_{10}\). Tại pixel \((i,j)\) trong block, thay đổi trường \(F\) là

\[
\Delta F_{ij}=\delta_b\left(B_{01}(i,j)-B_{10}(i,j)\right).
\]

Update RGB norm nhỏ nhất:

\[
\boxed{
\Delta\mathbf x_{ij}
=\Delta F_{ij}\,
\frac{\mathbf g}{\|\mathbf g\|_2^2}.
}
\]

Sau đó code:

1. clip về \([0,255]\);
2. round sang `uint8`;
3. chạy **integer-lattice closure** để bảo đảm parity trên ảnh số nguyên cuối cùng vẫn đúng.

Closure không thay đổi luật QIM; nó sửa sai số do lượng tử pixel 8-bit.

---

## 9. Luật trích xuất của DCT–QR

### 9.1 Đồng bộ

Nếu ảnh bị dịch/biến đổi nhẹ, proposal dùng pilot và certificate để chọn alignment. Đây là bước trước giải mã payload.

### 9.2 Hiệu chỉnh gain bằng QR carrier-\(r_{11}\)

Với ma trận phân tích không lift, QR canonical cho

\[
A_b=Q_bR_b.
\]

Scale được dùng là

\[
s_b=|(R_b)_{11}|.
\]

Key lưu reference \(s_b^0\) lấy từ ảnh watermarked cuối cùng. Trên ảnh bị attack:

\[
\alpha_b=
\operatorname{clip}\left(
\frac{s_b}{s_b^0},
\alpha_{\min},\alpha_{\max}
\right).
\]

Carrier hiệu chỉnh:

\[
\boxed{
\widetilde v_b=
\frac{v_b^{(a)}}{\alpha_b^{\gamma}}.
}
\]

Default public DCT–QR dùng \(\gamma=0.90\).

### 9.3 Quyết định parity

\[
q_b=\operatorname{round}\left(\frac{\widetilde v_b}{\Delta_b}\right),
\]

\[
\boxed{
\widehat c_b=q_b\bmod2.
}
\]

Confidence QIM trong code:

\[
\operatorname{conf}_b=
\operatorname{clip}
\left(
1-2\left|
\frac{\widetilde v_b}{\Delta_b}-q_b
\right|,
0,1
\right).
\]

### 9.4 XOR ngược coset

Key lưu packed flip mask của từng payload bit. Vì

\[
c_b=u_b\oplus s_j^*,
\]

nên

\[
\boxed{
\widehat u_b=\widehat c_b\oplus s_j^*.
}
\]

Sau đó inverse Arnold đưa bit về đúng vị trí watermark ban đầu.

Một tính chất quan trọng:

\[
\mathbf1[\widehat u_b\ne u_b]
=
\mathbf1[\widehat c_b\ne c_b].
\]

Nghĩa là XOR ngược coset **không tự tạo thêm lỗi bit**.

---

## 10. Pseudocode DCT–QR

```text
NHÚNG
1. W -> Arnold scramble -> u
2. Tính trường F và DCT 8x8
3. Tính QR certificate + reliability cho từng block
4. Chọn Delta_b theo reliability
5. Sort block theo reliability, ghép từng cặp G_j
6. Với mỗi cặp:
      tính cost s=0
      tính cost s=1
      s_j* = argmin(cost)
      c_b = u_b XOR s_j*
7. v_b* = QIM projection P_c(v_b)
8. Update RGB theo hướng minimum-norm
9. Integer-lattice closure
10. Lưu schedule, Delta_b, coset flip-mask, gain reference vào key

TRÍCH XUẤT
1. Đồng bộ ảnh bằng pilot/certificate
2. Tính carrier v_b^(a)
3. Tính QR carrier-r11 scale và alpha_b
4. v_tilde = v_b^(a) / alpha_b^gamma
5. q = round(v_tilde / Delta_b)
6. c_hat = q mod 2
7. u_hat = c_hat XOR s_j*
8. Inverse Arnold
9. Nếu evidence không đủ chắc chắn: MAP refinement
```

---

# PHẦN II — DCT–SCHUR SP-SCQIM

## 11. Điểm cốt lõi của `dct_schur_rescue`

Implementation hiện tại **không phải** “Schur rescue phụ trợ cho DCT–QR”. Nó là proposal độc lập:

> **Spectrum-Preserving Schur Coupling QIM (SP-SCQIM)**.

Source chính:

```text
src/qr64_certified/proposals/schur_coupling_qim.py
```

Mỗi block DCT chọn 6 hệ số làm vector strict-upper Schur ảo:

\[
\boxed{
n=
[n_{12},n_{13},n_{14},n_{23},n_{24},n_{34}]^T\in\mathbb R^6.
}
\]

Các vị trí DCT tương ứng trong code:

```text
(0,1), (1,0), (0,2), (1,1), (2,0), (1,2)
```

---

## 12. Ba hướng coupling trực giao

Code định nghĩa ba vector rồi chuẩn hóa thành unit norm:

\[
h_1\propto[1,-1,0,0,0,0],
\]

\[
h_2\propto[1,1,-1,-1,0,0],
\]

\[
h_3\propto[1,1,1,1,-2,-2].
\]

Gọi

\[
H=
\begin{bmatrix}
h_1^T\\h_2^T\\h_3^T
\end{bmatrix}
\in\mathbb R^{3\times6}.
\]

Sau chuẩn hóa:

\[
\boxed{HH^T=I_3.}
\]

Ba carrier của một block là

\[
y=Hn,
\qquad y_k=h_k^Tn.
\]

---

## 13. Tạo ba bản sao payload xen kẽ

Watermark được Arnold-scramble trước. Sau đó code sinh ba permutation độc lập từ cùng seed:

\[
\pi_1,\pi_2,\pi_3.
\]

Mỗi carrier \(k\) mang một bản permutation của payload:

\[
b_{k,b}=u_{\pi_k(b)}.
\]

Vì vậy một bit watermark có thể nhận evidence từ ba coupling khác nhau sau khi đảo permutation.

---

## 14. Target parity trong Schur coupling

Với carrier \(y_k\), step \(\Delta\) và bit \(b_k\), code chọn integer \(q_k\) gần nhất sao cho

\[
q_k\bmod2=b_k.
\]

Target:

\[
\boxed{
t_k=q_k\Delta.}
\]

Khác DCT–QR, SP-SCQIM ở implementation này dùng **nearest parity center**, không dùng QIM interval guard như `dct_qr`.

---

## 15. Luật nhúng Schur: nghiệm đóng-form minimum Frobenius

Ta cần tìm \(n^*\) gần \(n\) nhất nhưng thỏa

\[
Hn^*=t.
\]

Bài toán:

\[
\min_{n^*}\|n^*-n\|_2^2
\quad\text{s.t.}\quad Hn^*=t.
\]

Do các hàng của \(H\) trực chuẩn, nghiệm chiếu trực giao là

\[
\boxed{
n^*=n+H^T(t-Hn).
}
\]

Đây chính là luật nhúng được code thực thi.

### 15.1 Chứng minh ngắn

Đặt \(\delta=n^*-n\). Ràng buộc trở thành

\[
H\delta=t-Hn=r.
\]

Vì \(HH^T=I\), một nghiệm là

\[
\delta_0=H^Tr.
\]

Mọi nghiệm khác có dạng

\[
\delta=H^Tr+z,
\qquad Hz=0.
\]

Không gian hàng của \(H\) trực giao với null-space của \(H\), nên

\[
\|\delta\|_2^2
=\|H^Tr\|_2^2+\|z\|_2^2
\ge\|H^Tr\|_2^2.
\]

Dấu bằng khi \(z=0\). Vậy \(n^*=n+H^T(t-Hn)\) là nghiệm duy nhất có norm thay đổi nhỏ nhất.

---

## 16. Vì sao spectrum, trace và determinant được bảo toàn?

Code tạo ma trận tam giác trên 4×4:

\[
T=
\begin{bmatrix}
\lambda_1&n_{12}&n_{13}&n_{14}\\
0&\lambda_2&n_{23}&n_{24}\\
0&0&\lambda_3&n_{34}\\
0&0&0&\lambda_4
\end{bmatrix}.
\]

Luật nhúng chỉ thay **strict-upper entries** \(n_{ij}\), không đổi diagonal \(\lambda_i\).

Với ma trận tam giác:

\[
\operatorname{spec}(T)=\{\lambda_1,\lambda_2,\lambda_3,\lambda_4\},
\]

\[
\operatorname{tr}(T)=\sum_i\lambda_i,
\]

\[
\det(T)=\prod_i\lambda_i.
\]

Do đó, ở miền floating-point trước khi quay lại ảnh `uint8`:

\[
\boxed{
\text{spectrum, trace, determinant không đổi.}
}
\]

Đây là ý nghĩa toán học của tên **Spectrum-Preserving**.

---

## 17. Chuyển thay đổi Schur coupling về ảnh

Sau khi viết \(n^*\) vào 6 hệ số DCT:

1. IDCT block;
2. lấy chênh lệch trường \(\Delta F\);
3. chiếu \(\Delta F\) sang RGB theo hướng minimum-norm \(\mathbf g/\|\mathbf g\|^2\);
4. round và clip về `uint8`;
5. kiểm tra lại ba bản sao vật lý;
6. nếu chưa exact thì closure re-project từ ảnh số nguyên hiện tại.

---

## 18. Luật trích xuất SP-SCQIM

### 18.1 Tính lại ba carrier

Từ ảnh cần kiểm tra, lấy vector coupling \(n^{(a)}\). Với mỗi \(k\):

\[
y_k^{(a)}=h_k^Tn^{(a)}.
\]

### 18.2 Gain normalization

Code dùng một scale phổ cục bộ từ các hệ số DCT riêng biệt:

\[
s_b=\sqrt{\operatorname{mean}(c_{b,j}^2)+10^{-6}}.
\]

Key giữ reference \(s_b^0\) từ ảnh watermarked cuối cùng. Ratio được median-filter 3×3 và clip:

\[
\alpha_b=\operatorname{clip}
\left(\frac{s_b}{s_b^0},0.55,1.45\right).
\]

Carrier hiệu chỉnh:

\[
\boxed{
\widetilde y_{k,b}=
\frac{y_{k,b}^{(a)}}{\alpha_b^{\gamma}}.
}
\]

### 18.3 Quyết định bit cho từng copy

\[
q_{k,b}=\operatorname{round}
\left(\frac{\widetilde y_{k,b}}{\Delta}\right),
\]

\[
\boxed{
\widehat b_{k,b}=q_{k,b}\bmod2.
}
\]

Confidence:

\[
\operatorname{conf}_{k,b}
=\operatorname{clip}
\left(
1-2\left|
\frac{\widetilde y_{k,b}}{\Delta}-q_{k,b}
\right|,0,1
\right).
\]

### 18.4 Đưa ba copy về cùng payload index

Mỗi copy được inverse-permute bởi \(\pi_k\), sau đó đổi bit thành evidence có dấu:

\[
e_{k,i}=\operatorname{sign}(\widehat b_{k,i})
\left(c_0+c_1\operatorname{conf}_{k,i}^{p}\right).
\]

Tổng evidence:

\[
\boxed{
E_i=\sum_{k=1}^3e_{k,i}.
}
\]

Sau inverse Arnold, code có thể dùng ICM/MAP để tận dụng tính liên tục không gian của watermark nhị phân.

---

## 19. Pseudocode DCT–Schur SP-SCQIM

```text
NHÚNG
1. W -> Arnold scramble
2. Sinh 3 permutation pi1, pi2, pi3
3. Với mỗi 8x8 block:
      lấy 6 DCT coefficient -> n in R^6
      y = H n
      với k=1..3:
          chọn q_k gần nhất, parity(q_k)=bit_k
          t_k = q_k * Delta
      n* = n + H^T(t - Hn)
      ghi n* về DCT block
4. IDCT + minimum-norm RGB update
5. uint8 closure nếu cần
6. Lưu spectral gain reference vào key

TRÍCH XUẤT
1. Tạo candidate image nếu candidate-search bật
2. Với từng candidate:
      lấy n
      gain normalize
      y_k = h_k^T n
      q_k = round(y_k/Delta)
      bit_k = q_k mod 2
      inverse permutation cho 3 copy
      tính agreement + confidence + evidence
3. Chọn candidate có score cao nhất
4. Cộng evidence ba copy
5. Inverse Arnold
6. ICM/MAP -> watermark cuối
```

---

# PHẦN III — SPATIAL NORMALIZED-RESIDUAL DetQR

## 20. Ý tưởng của `spatial_cd_detqr`

Phương pháp này **không dùng DCT** để nhúng payload. Nó làm việc trực tiếp trong block luminance 8×8.

Source chính:

```text
src/qr64_certified/proposals/cd_detqr.py
```

Mỗi block dùng các pattern Hadamard \(P\in\{-1,+1\}^{8\times8}\). Với một pattern:

- \(u\): trung bình luminance tại các pixel \(P=+1\);
- \(v\): trung bình luminance tại các pixel \(P=-1\).

Tạo ma trận

\[
\boxed{
A=
\begin{bmatrix}
u&1\\
v&1
\end{bmatrix}.
}
\]

---

## 21. Determinant và normalized QR residual

Ta có trực tiếp

\[
\boxed{\det(A)=u-v.}
\]

Đặt

\[
d=u-v,
\qquad k=u+v.
\]

Từ QR canonical \(A=QR\):

\[
r_{11}=\sqrt{u^2+v^2}.
\]

Carrier chuẩn hóa trong code:

\[
\boxed{
z(A)=\det(Q)r_{22}
=\frac{\det(A)}{r_{11}}
=\frac{u-v}{\sqrt{u^2+v^2}}.
}
\]

Dùng \(k,d\):

\[
u=\frac{k+d}{2},\qquad
v=\frac{k-d}{2},
\]

nên

\[
u^2+v^2=\frac{k^2+d^2}{2}.
\]

Do đó

\[
\boxed{
z(A)=\frac{\sqrt2\,d}{\sqrt{k^2+d^2}}.}
\]

---

## 22. Mã hóa bit bằng dấu

Watermark bit \(w\) trước hết có thể XOR với pseudorandom payload mask:

\[
c=w\oplus m.
\]

Ánh xạ sang dấu:

\[
\boxed{
s=\begin{cases}
+1,&c=1,\\
-1,&c=0.
\end{cases}}
\]

Mục tiêu là làm cho determinant cuối cùng mang dấu \(s\), đồng thời có biên đủ lớn.

---

## 23. Update antisymmetric trong không gian ảnh

Với amplitude nguyên \(a\ge0\), code update block RGB:

\[
\boxed{
I^*=I+s\,a\,P
}
\]

trên cả ba kênh, sau đó clip về `[0,255]`.

Vì nhóm \(P=+1\) tăng \(sa\), nhóm \(P=-1\) giảm \(sa\):

\[
u'=u+sa,
\]

\[
v'=v-sa.
\]

Suy ra

\[
\boxed{k'=u'+v'=k}
\]

và

\[
\boxed{d'=u'-v'=d+2sa.}
\]

Đây là điểm rất đẹp của luật nhúng: **tổng \(u+v\) được giữ, chỉ hiệu \(u-v\) được điều khiển**.

---

## 24. Ràng buộc QR margin và determinant khác 0

Proposal yêu cầu normalized margin

\[
\boxed{s\,z(A')\ge\mu}
\]

và determinant safety

\[
\boxed{|\det(A')|=|d'|\ge\delta.}
\]

Vì dấu mong muốn là \(s\), điều kiện thứ hai có thể kết hợp thành ngưỡng cho \(sd'\).

Từ

\[
s\frac{\sqrt2d'}{\sqrt{k^2+d'^2}}\ge\mu,
\]

với \(0<\mu<\sqrt2\), bình phương và biến đổi cho ta

\[
\boxed{
sd'\ge
\frac{\mu|k|}{\sqrt{2-\mu^2}}.}
\]

Kết hợp với determinant margin \(\delta\):

\[
\boxed{
T=\max\left(
\delta,
\frac{\mu|k|}{\sqrt{2-\mu^2}}
\right).
}
\]

Cần

\[
sd'=s(d+2sa)=sd+2a\ge T.
\]

---

## 25. Luật nhúng đóng-form của DetQR

Từ

\[
sd+2a\ge T,
\]

suy ra

\[
a\ge\frac{T-sd}{2}.
\]

Do \(a\) phải là số nguyên không âm, biên độ nhỏ nhất là

\[
\boxed{
a^*=\max\left(
0,
\left\lceil\frac{T-sd}{2}\right\rceil
\right).
}
\]

Thay \(T\):

\[
\boxed{
a^*=\max\left(
0,
\left\lceil
\frac{
\max\left(\delta,\mu|u+v|/\sqrt{2-\mu^2}\right)
-s(u-v)}{2}
\right\rceil
\right).
}
\]

Đây chính là hàm `minimum_integer_amplitude()` trong code.

### Ý nghĩa

- Nếu carrier đã đúng dấu và đủ xa biên: \(a^*=0\), block không bị sửa.
- Nếu chưa đủ: chỉ thêm **số mức nguyên nhỏ nhất** cần thiết.
- Điều này trực tiếp tối ưu imperceptibility ở mức local amplitude.

---

## 26. Chọn pattern có distortion thấp nhất

Có tối đa 10 pattern Hadamard ứng viên. Với mỗi pattern \(p\), code tính \(a_p^*\) rồi dùng cost

\[
\boxed{
J_p=(a_p^*)^2
+\lambda_B B_p
-10^{-6}s z_p.
}
\]

Trong đó:

- \(B_p\): boundary cost của pattern;
- \(\lambda_B\): `boundary_penalty`;
- thành phần rất nhỏ \(-10^{-6}s z_p\) dùng tie-break theo hướng carrier thuận lợi.

Chọn

\[
\boxed{p^*=\arg\min_pJ_p.}
\]

Sau đó nhúng bằng \(a_{p^*}^*\).

---

## 27. Pilot của Spatial DetQR

Pilot dùng **cùng QR/determinant domain**, không phải một carrier DCT riêng.

Trong ảnh host ban đầu, code tìm các carrier pilot có determinant âm đủ an toàn, rồi nhúng chúng về giả thuyết dương bằng cùng công thức amplitude đóng-form.

Khi extract:

\[
\text{pilot detected}
\Longleftrightarrow
\#\{\det(A_{pilot})>0\}>rac{N_p}{2}.
\]

Pilot cũng được dùng để đánh giá rotation/shear candidate phục vụ affine synchronization.

---

## 28. Luật trích xuất Spatial DetQR

Sau alignment:

1. dùng pattern đã lưu trong key;
2. tính \(u,v\) và determinant;
3. đọc coded bit bằng dấu:

\[
\boxed{
\widehat c=
\begin{cases}
1,&u-v>0,\\
0,&u-v\le0.
\end{cases}}
\]

4. kiểm tra majority pilot;
5. nếu pilot được phát hiện và payload mask bật:

\[
\boxed{\widehat w=\widehat c\oplus m.}
\]

Nếu pilot không được phát hiện, implementation hiện tại giữ `coded` payload thay vì unmask.

Điểm khác DCT–QR/Schur: **không có bước `round(carrier/Delta) mod 2`**. Bit ở đây nằm trong **dấu của determinant**.

---

## 29. Pseudocode Spatial DetQR

```text
NHÚNG
1. W -> bits
2. Sinh payload mask m; coded = bits XOR m
3. Với mỗi block và mỗi Hadamard pattern:
      tính u, v
      d = u-v, k=u+v
      s = +1 nếu coded=1, ngược lại -1
      T = max(delta, mu|k|/sqrt(2-mu^2))
      a* = max(0, ceil((T-sd)/2))
4. Chọn pattern có cost nhỏ nhất
5. I* = I + s a* P
6. Nhúng pilot cùng miền determinant
7. Closure để khôi phục exact clean extraction sau clip/round
8. Lưu pattern schedule, pilot schedule, mask seed vào key

TRÍCH XUẤT
1. Dùng pilot để thử affine alignment
2. Tính determinant tại payload patterns
3. c_hat = 1 nếu determinant > 0, ngược lại 0
4. Majority vote pilot
5. Nếu pilot detected: w_hat = c_hat XOR mask
6. Reshape thành 64x64
```

---

# PHẦN IV — SO SÁNH TOÁN HỌC

## 30. So sánh ba luật nhúng

| Thành phần | DCT–QR | DCT–Schur SP-SCQIM | Spatial DetQR |
|---|---|---|---|
| Domain | DCT | DCT / Schur coordinates | Spatial luminance |
| Carrier | \((C_{01}-C_{10})/2\) | \(h_k^Tn\), \(k=1,2,3\) | \(d=u-v=\det(A)\) |
| Kiểu bit | parity lattice | parity lattice | sign determinant |
| Luật tối ưu | chọn coset \(s_j\) có projection energy nhỏ nhất | orthogonal projection minimum Frobenius | integer amplitude nhỏ nhất thỏa margin |
| Công thức chính | \(s_j^*=\arg\min_s\sum|P_{u\oplus s}(v)-v|^2\) | \(n^*=n+H^T(t-Hn)\) | \(a^*=\max(0,\lceil(T-sd)/2\rceil)\) |
| Redundancy | 1 physical bit/block + coset side info | 3 coupling copies | 1 determinant bit/block |
| Gain correction | QR carrier-\(r_{11}\) | spectral DCT reference | không dùng gain-QIM |
| Extract decision | round + parity | round + parity + vote | sign of determinant |
| Blindness | key-assisted blind | key-assisted blind | key-assisted blind |

---

## 31. Khi nào một bit được xem là “đúng”?

### DCT–QR

Sau gain correction:

\[
\operatorname{round}(\widetilde v/\Delta)\bmod2=c.
\]

Sau đó XOR coset để về bit watermark.

### DCT–Schur

Cho từng copy:

\[
\operatorname{round}(\widetilde y_k/\Delta)\bmod2=b_k,
\]

rồi ba copy được hợp nhất bằng evidence/vote.

### Spatial DetQR

\[
\operatorname{sign}(u-v)=s.
\]

Không cần QIM parity.

---

# PHẦN V — CÁCH CHẠY REPOSITORY

## 32. Cài đặt trên Linux / WSL / macOS

Đứng tại thư mục gốc `Hanoi-main`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JILP_NUM_THREADS=1
```

Để giảm khác biệt runtime từ BLAS/OpenMP, có thể thêm:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## 33. Cài đặt trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src;$PWD\scripts"
$env:JILP_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

---

## 34. Kiểm tra ba proposal có được nhận diện không

```bash
python scripts/list_proposals.py
```

Kết quả phải liệt kê đúng:

```text
dct_qr
dct_schur_rescue
spatial_cd_detqr
```

---

## 35. Smoke test nhanh nhất cho cả ba proposal

```bash
python scripts/smoke_test_proposals.py
```

Khi chạy trực tiếp trên ZIP đã cung cấp trong quá trình tạo tài liệu này, kết quả clean trên `lenna.bmp` là:

| Method | PSNR smoke | Clean NC | det_nonzero |
|---|---:|---:|---|
| `dct_qr` | 50.2777 dB | 1.0000 | true |
| `dct_schur_rescue` | 48.1566 dB | 1.0000 | true |
| `spatial_cd_detqr` | 55.8722 dB | 1.0000 | true |

Đây là **smoke result clean**, không phải kết luận benchmark attacked-NC cuối cùng.

---

## 36. Test trực tiếp phần core của 3 proposal

```bash
pytest -q \
  tests/test_dct_qr_pairwise_coset.py \
  tests/test_direct_schur_rescue.py \
  tests/test_cd_detqr.py
```

Kết quả đã kiểm tra trên package hiện tại:

```text
15 passed
```

### Lưu ý về `pytest -q` toàn repo

Hiện tại lệnh:

```bash
pytest -q
```

dừng ở collection vì file cũ `tests/test_pso.py` vẫn import:

```python
particle_swarm_maximize
```

trong khi optimizer hiện tại đã chuyển sang ABC và `qr64_certified.optimization` không export hàm PSO đó. Vì vậy dòng README nói `49 passed` **không tái lập được với trạng thái ZIP hiện tại** nếu chạy nguyên lệnh `pytest -q`.

---

# PHẦN VI — LỆNH CHẠY TỪNG PROPOSAL

## 37. DCT–QR: chạy nhanh 1 host

Validation riêng của pairwise coset:

```bash
python scripts/validate_dct_qr_pairwise_coset.py --host-limit 1
```

Full tất cả BMP trong `data/host`:

```bash
python scripts/validate_dct_qr_pairwise_coset.py
```

Chỉ định output:

```bash
python scripts/validate_dct_qr_pairwise_coset.py \
  --output results/my_dct_qr_validation.json
```

---

## 38. DCT–QR: benchmark dùng config ABC hiện tại

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --stage after_abc \
  --save-images
```

In config thật sự sau khi áp dụng flag:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --stage after_abc \
  --print-effective-config
```

Ví dụ ablation bỏ coset optimization:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --ablation no_coset_optimization \
  --gain-gamma 0.90 \
  --step 13.25
```

---

## 39. DCT–Schur SP-SCQIM

Reproduce validation riêng với step 9.0:

```bash
python scripts/benchmark_schur_sp_scqim.py \
  --step 9.0 \
  --output results/schur_sp_scqim/reproduced.json
```

Chỉ chạy một số host theo tên:

```bash
python scripts/benchmark_schur_sp_scqim.py \
  --hosts lenna baboon peppers \
  --step 9.0
```

Benchmark theo config `after_abc`:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_schur_rescue \
  --stage after_abc \
  --save-images
```

---

## 40. Spatial DetQR

Benchmark 1 host với config ABC:

```bash
python scripts/run_proposal_benchmark.py \
  --method spatial_cd_detqr \
  --stage after_abc \
  --save-images
```

Chạy redesigned validation chỉ cho Spatial DetQR trên 1 host:

```bash
python scripts/run_redesigned_validation.py \
  --methods spatial_cd_detqr \
  --host-limit 1
```

Chạy tất cả host:

```bash
python scripts/run_redesigned_validation.py \
  --methods spatial_cd_detqr
```

---

## 41. Chạy cả ba proposal trên 1 host

```bash
python scripts/run_redesigned_validation.py --host-limit 1
```

Chạy full host set:

```bash
python scripts/run_redesigned_validation.py
```

Lệnh này đọc:

```text
configs/dct_qr_after_abc.json
configs/dct_schur_rescue_after_abc.json
configs/spatial_cd_detqr_after_abc.json
```

và mặc định fail nếu clean NC của bất kỳ host nào nhỏ hơn \(1-10^{-12}\).

---

## 42. Chạy ablation

DCT–QR:

```bash
python scripts/run_ablation_study.py --method dct_qr
```

DCT–Schur:

```bash
python scripts/run_ablation_study.py --method dct_schur_rescue
```

Spatial DetQR:

```bash
python scripts/run_ablation_study.py --method spatial_cd_detqr
```

Chỉ chạy clean để kiểm tra flag nhanh:

```bash
python scripts/run_ablation_study.py \
  --method dct_qr \
  --clean-only
```

---

## 43. Tối ưu tham số bằng Artificial Bee Colony (ABC)

Full cho cả ba:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 20 \
  --cycles 20 \
  --seed 2026
```

Chỉ DCT–QR:

```bash
python scripts/optimize_parameters.py \
  --method dct_qr \
  --food-sources 20 \
  --cycles 20 \
  --seed 2026
```

Smoke optimization rất nhanh:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 2 \
  --cycles 1 \
  --onlookers 1 \
  --attack-limit 1 \
  --config-output-dir results/abc_smoke_configs
```

Không nên dùng smoke optimization này để báo cáo paper result; nó chỉ kiểm tra pipeline.

---

# PHẦN VII — API PYTHON TỐI THIỂU

## 44. Gọi proposal bằng Python

```python
from qr64_certified.proposals import embed_proposal, extract_proposal

# host: numpy uint8, shape (512, 512, 3)
# watermark: binary/uint8, shape (64, 64)

watermarked, key = embed_proposal("dct_qr", host, watermark)
recovered = extract_proposal(watermarked, key)
```

Đổi phương pháp:

```python
watermarked, key = embed_proposal("dct_schur_rescue", host, watermark)
recovered = extract_proposal(watermarked, key)
```

```python
watermarked, key = embed_proposal("spatial_cd_detqr", host, watermark)
recovered = extract_proposal(watermarked, key)
```

Nếu cần metadata:

```python
watermarked, key, embed_meta = embed_proposal(
    "dct_schur_rescue",
    host,
    watermark,
    return_metadata=True,
)

recovered, extract_meta = extract_proposal(
    watermarked,
    key,
    return_metadata=True,
)
```

---

# PHẦN VIII — CÁCH TRÌNH BÀY TRONG PAPER/THESIS

## 45. Phát biểu ngắn gọn luật nhúng của từng proposal

### DCT–QR

> Với mỗi cặp block gần nhau về QR reliability, proposal chọn một nhãn coset nhị phân chung sao cho tổng năng lượng chiếu QIM của cả cặp là nhỏ nhất, sau đó thực hiện minimum-distortion guarded-QIM projection và integer-lattice closure trên ảnh RGB 8-bit.

Công thức trung tâm:

\[
\boxed{
s_j^*=\arg\min_s\sum_{b\in G_j}|P_{u_b\oplus s}(v_b)-v_b|^2.}
\]

### DCT–Schur SP-SCQIM

> Ba tổ hợp trực giao của sáu strict-upper Schur couplings được ép vào ba lattice parity target. Do basis coupling có các hàng trực chuẩn, nghiệm nhúng có biến dạng Frobenius nhỏ nhất được cho trực tiếp bởi phép chiếu trực giao.

Công thức trung tâm:

\[
\boxed{n^*=n+H^T(t-Hn).}
\]

### Spatial DetQR

> Mỗi bit được mã hóa bằng dấu của determinant trong một ma trận QR 2×2 hình thành từ hai trung bình luminance đối nghịch. Biên độ spatial nguyên nhỏ nhất được suy ra đóng-form để đồng thời thỏa normalized QR margin và determinant safety.

Công thức trung tâm:

\[
\boxed{
a^*=\max\left(0,\left\lceil\frac{T-sd}{2}\right\rceil\right).}
\]

---

## 46. Phát biểu ngắn gọn luật trích xuất

### DCT–QR

\[
\boxed{
\widehat u_b=
\left[
\operatorname{round}
\left(
\frac{v_b^{(a)}}{\alpha_b^\gamma\Delta_b}
\right)
\bmod2
\right]
\oplus s_j^*.
}
\]

### DCT–Schur

Cho ba carrier:

\[
\widehat b_{k,b}=
\operatorname{round}
\left(
\frac{h_k^Tn_b^{(a)}}{\alpha_b^\gamma\Delta}
\right)\bmod2,
\]

sau đó inverse permutation và fusion ba evidence.

### Spatial DetQR

\[
\boxed{
\widehat c_b=\mathbf1[\det(A_b)>0],
\qquad
\widehat w_b=\widehat c_b\oplus m_b
}
\]

khi pilot được phát hiện.

---

## 47. Các lưu ý khoa học quan trọng

1. **Key-assisted blind không đồng nghĩa không có side information.** Cả ba phương pháp không cần original host khi extract, nhưng key vẫn chứa schedule/reference cần thiết.
2. Với DCT–QR, packed coset mask là một phần thiết yếu của luật giải mã; không nên ẩn chi tiết này khi viết paper.
3. Với Schur, bảo toàn spectrum/trace/determinant là tính chất của constructed triangular Schur matrix ở miền floating-point; sau IDCT và lượng tử `uint8`, cần phân biệt invariant theorem với ảnh vật lý cuối cùng.
4. Với Spatial DetQR, điều kiện đóng-form được suy ra trước clip; closure được dùng để phục hồi điều kiện trên ảnh số nguyên cuối.
5. Kết quả smoke clean NC=1 không chứng minh robustness dưới attack. Báo cáo paper nên tách rõ clean exactness, attacked NC, worst-case NC, PSNR/SSIM, latency và key size.
6. Khi so sánh latency, phải cố định số thread (`JILP_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) và cùng hardware.

---

## 48. File source cần đọc khi chỉnh sửa proposal

```text
# DCT–QR
src/qr64_certified/proposals/method.py
src/qr64_certified/proposals/config.py
src/qr64_certified/proposals/certificate.py
src/qr64_certified/_jilp_core/engine.py
configs/dct_qr_after_abc.json

# DCT–Schur SP-SCQIM
src/qr64_certified/proposals/schur_coupling_qim.py
configs/dct_schur_rescue_after_abc.json
configs/dct_schur_sp_scqim.json

# Spatial DetQR
src/qr64_certified/proposals/cd_detqr.py
configs/spatial_cd_detqr_after_abc.json

# Runner / evaluation
scripts/run_proposal_benchmark.py
scripts/run_redesigned_validation.py
scripts/run_ablation_study.py
scripts/optimize_parameters.py
scripts/smoke_test_proposals.py
```

---

## 49. Kết luận

Ba proposal giải ba bài toán tối ưu local khác nhau:

\[
\boxed{
\text{DCT–QR: tối thiểu hóa QIM projection energy qua coset.}
}
\]

\[
\boxed{
\text{DCT–Schur: tối thiểu hóa Frobenius update dưới ràng buộc coupling parity.}
}
\]

\[
\boxed{
\text{Spatial DetQR: tối thiểu hóa integer amplitude dưới QR/determinant margin.}
}
\]

Nếu cần mô tả ngắn nhất về “luật nhúng” của cả hệ thống, có thể nói:

> **DCT–QR tối ưu cách gán parity trước khi chiếu QIM; DCT–Schur giải trực tiếp phép chiếu trực giao tối thiểu lên ba ràng buộc parity; Spatial DetQR suy ra biên độ nguyên nhỏ nhất để ép dấu determinant và biên QR. Luật trích xuất tương ứng lần lượt là parity sau gain correction + XOR coset, parity fusion của ba Schur coupling, và quyết định dấu determinant + unmask bằng pilot.**
