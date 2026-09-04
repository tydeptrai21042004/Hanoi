# Đề xuất `spatial_qr`: QR không gian + pairwise coset + gain normalization, không DCT

## 1. Mục tiêu

`spatial_qr` là đối chứng trực tiếp với `dct_qr`, nhưng **loại bỏ hoàn toàn DCT/IDCT**. Phương pháp làm việc trực tiếp trên luminance của các block 8×8.

Chuỗi xử lý:

\[
RGB \rightarrow Y \rightarrow \text{block }8\times8
\rightarrow \text{spatial carrier + QR certificate}
\rightarrow \text{pairwise coset QIM}
\rightarrow RGB.
\]

Không có bước biến đổi DCT trong nhúng hoặc trích xuất.

## 2. Spatial carrier

Với block luminance \(Y_b\), carrier được chọn là

\[
v_b=\frac{Y_b(1,2)-Y_b(2,1)}{2}.
\]

Hai vị trí này nằm ngoài ma trận QR trung tâm 4×4, vì vậy cập nhật payload không ghi đè trực tiếp lên QR certificate.

Nếu muốn thay đổi carrier một lượng \(\delta_b\), cập nhật đối xứng

\[
Y_b(1,2)\leftarrow Y_b(1,2)+\delta_b,
\qquad
Y_b(2,1)\leftarrow Y_b(2,1)-\delta_b
\]

cho đúng

\[
v_b' = v_b+\delta_b.
\]

## 3. QR certificate trong miền không gian

Lấy patch luminance trung tâm 4×4, trừ mean của chính patch, rồi thêm regularization ảo:

\[
A_b=P_b-\bar P_b\mathbf 1+\lambda I,
\qquad A_b=Q_bR_b.
\]

QR có hai vai trò:

1. **reliability scheduling**: dùng một chỉ số ổn định dựa trên diagonal của \(R_b\) để gán ba mức QIM;
2. **gain reference**: dùng \(|R_{11}|\) làm scale reference khi extract.

## 4. Pairwise coset optimization

Sau khi sắp block theo QR reliability, các block được ghép thành cặp. Với mỗi nhóm \(G_j\), chọn một coset bit chung

\[
s_j^*=\arg\min_{s\in\{0,1\}}
\sum_{b\in G_j}\left|P_{u_b\oplus s}(v_b)-v_b\right|^2.
\]

Do \(s=0\) luôn là phương án hợp lệ,

\[
D_{\mathrm{optimized}}\le D_{\mathrm{original}}.
\]

Đây là cùng nguyên lý tối thiểu distortion của `dct_qr`, nhưng carrier nằm hoàn toàn trong miền không gian.

## 5. Gain normalization

Khi extract, tính

\[
\widehat\alpha_b=
\frac{|R_{11,b}^{(a)}|}{|R_{11,b}^{(w)}|+\varepsilon},
\]

và chuẩn hóa

\[
\widetilde v_b=
\frac{v_b^{(a)}}{\widehat\alpha_b^{\gamma}}.
\]

Carrier là hiệu hai pixel nên additive brightness gần như triệt tiêu; QR reference hỗ trợ bù multiplicative gain/contrast.

## 6. Trạng thái hiện tại

- Không dùng DCT/IDCT: **đã kiểm tra bằng source test và metadata**.
- Clean round-trip trên **13/13 host**: **NC = 1.0**; mean PSNR ≈ **51.81 dB**, minimum PSNR ≈ **51.54 dB**.
- Default Lenna PSNR ≈ **51.94 dB**.
- Brightness/contrast ±10% trên Lenna giữ NC xấp xỉ **0.9993–0.9998**.
- Trên 15 moderate attacks của Lenna: mean NC ≈ **0.9141**, minimum NC ≈ **0.7148**. Vì vậy robustness vẫn thấp hơn `dct_qr`; chưa được claim publication-validated.
- Một host có spatial QR certificate gần singular theo `det_epsilon`; determinant **không** phải hard constraint của method này (khác `spatial_cd_detqr`).

## 7. Code

- `src/qr64_certified/proposals/spatial_qr.py`
- `src/qr64_certified/proposals/spatial_qr_common.py`
- `configs/spatial_qr_before.json`
- `configs/spatial_qr_after_abc.json`
- `tests/test_spatial_qr_family.py`
