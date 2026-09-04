# Đề xuất `spatial_qr_direct_r`: Direct-R QIM trong miền không gian, không DCT

## 1. Pipeline

\[
RGB\rightarrow Y\rightarrow\text{block }8\times8
\rightarrow A_b\in\mathbb R^{4\times4}
\rightarrow QR
\rightarrow R'\rightarrow Q R'\rightarrow RGB.
\]

\(A_b\) là patch luminance trung tâm 4×4 với regularization ảo \(\lambda I\). Không có DCT hoặc IDCT.

## 2. Carrier

\[
v_b=\frac{R_{12}-R_{13}}{2}.
\]

Parity-QIM tạo target \(v_b^*\), đặt

\[
\delta_b=v_b^*-v_b.
\]

Cập nhật:

\[
R_{12}'=R_{12}+\delta_b,
\qquad
R_{13}'=R_{13}-\delta_b.
\]

Suy ra chính xác

\[
\frac{R_{12}'-R_{13}'}2=v_b^*.
\]

## 3. Tính chất minimum-Frobenius

Xét mọi cập nhật chỉ trên hai phần tử thỏa

\[
\Delta R_{12}-\Delta R_{13}=2\delta_b.
\]

Bài toán

\[
\min \left[(\Delta R_{12})^2+(\Delta R_{13})^2\right]
\]

có nghiệm

\[
\Delta R_{12}=\delta_b,
\qquad
\Delta R_{13}=-\delta_b.
\]

Do đó

\[
\|\Delta R\|_F=\sqrt2|\delta_b|.
\]

Ngoài ra cập nhật giữ nguyên

\[
R_{12}+R_{13},
\]

toàn bộ diagonal của \(R\), và vì vậy determinant của triangular factor trước quantization RGB cũng không đổi.

## 4. Trạng thái hiện tại

- Hoàn toàn không DCT/IDCT.
- Clean round-trip trên **13/13 host**: **NC = 1.0**; mean PSNR ≈ **50.73 dB**, minimum PSNR ≈ **50.49 dB**.
- Default Lenna PSNR ≈ **50.92 dB**.
- Trên 15 moderate attacks của Lenna: mean NC ≈ **0.8507**, minimum NC ≈ **0.3526**.
- Đây là proposal strict-blind với key nhỏ, nhưng JPEG Q70/median và một số hình học còn yếu; cần normalized carrier/gain compensation hoặc synchronization nếu phát triển thành phương pháp chính.
- Determinant không phải hard constraint; một host có central spatial matrix gần singular theo threshold hiện tại.

## 5. Code

- `src/qr64_certified/proposals/spatial_qr_direct_r.py`
- `src/qr64_certified/proposals/spatial_qr_common.py`
- `configs/spatial_qr_direct_r_before.json`
- `configs/spatial_qr_direct_r_after_abc.json`
- `tests/test_spatial_qr_family.py`
