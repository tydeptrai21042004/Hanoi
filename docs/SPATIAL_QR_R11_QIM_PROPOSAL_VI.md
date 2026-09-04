# Đề xuất `spatial_qr_r11_qim`: Direct-R11 QIM trong miền không gian, không DCT

## 1. Pipeline

\[
RGB\rightarrow Y\rightarrow\text{block }8\times8
\rightarrow A_b\in\mathbb R^{4\times4}
\rightarrow QR\rightarrow R_{11}\text{-QIM}
\rightarrow QR\text{ reconstruction}\rightarrow RGB.
\]

Ma trận QR là patch luminance trung tâm 4×4 cộng regularization ảo. Không dùng DCT/IDCT.

## 2. Carrier

Với canonical QR có diagonal dương,

\[
A_b=Q_bR_b,
\qquad R_{11}>0.
\]

Bit \(b\in\{0,1\}\) được nhúng bằng parity-QIM:

\[
R_{11}'=k^*\Delta,
\qquad k^*\bmod2=b,
\]

trong đó \(k^*\) là chỉ số quantization gần nhất thỏa parity yêu cầu.

Chỉ \(R_{11}\) được thay đổi tường minh; các phần tử khác của \(R\) giữ nguyên trước bước reconstruct.

## 3. Ý nghĩa khoa học

Phương pháp này là ablation trực tiếp nhất để trả lời câu hỏi:

> Nếu bỏ DCT và dùng chính norm/canonical first-column scale của QR trong miền không gian làm carrier, clean fidelity và robustness thay đổi thế nào?

Nó đơn giản hơn `spatial_qr` và `spatial_qr_direct_r`, nhưng raw \(R_{11}\) không bất biến theo global gain, vì vậy không nên mặc định kỳ vọng robustness brightness bằng phương pháp có gain normalization.

## 4. Trạng thái hiện tại

- Hoàn toàn không DCT/IDCT.
- Clean round-trip trên **13/13 host**: **NC = 1.0**; mean PSNR ≈ **50.58 dB**, minimum PSNR ≈ **50.44 dB**.
- Default Lenna PSNR ≈ **50.55 dB**.
- Closure có **saturation-aware same-parity downward rescue**: chỉ những block vẫn sai sau một uint8 closure round mới được phép chuyển xuống lattice point cùng parity cách 2 chỉ số; decoder không đổi. Cơ chế này sửa trường hợp host sáng mà không gây distortion diện rộng.
- Trên 15 moderate attacks của Lenna: mean NC ≈ **0.7059**, minimum NC ≈ **0.2220**; đây vẫn là method yếu nhất trong ba spatial QR mới.
- Nên xem đây là proposal/ablation mới, không phải robust main method. Determinant không phải hard constraint.

## 5. Code

- `src/qr64_certified/proposals/spatial_qr_r11_qim.py`
- `src/qr64_certified/proposals/spatial_qr_common.py`
- `configs/spatial_qr_r11_qim_before.json`
- `configs/spatial_qr_r11_qim_after_abc.json`
- `tests/test_spatial_qr_family.py`
