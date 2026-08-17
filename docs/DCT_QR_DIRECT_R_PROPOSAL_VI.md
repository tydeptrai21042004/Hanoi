# Proposal mới: DCT–QR Direct-R Differential QIM

## 1. Mục tiêu và khác biệt miền xử lý

Phương pháp `dct_qr_direct_r` được thêm từ ý tưởng QR trong tài liệu đầu vào,
nhưng QR **không được thực hiện trực tiếp trên khối ảnh miền không gian**.
Pipeline bắt buộc là:

\[
\text{RGB}\to Y\to \mathrm{DCT}_{8\times8}\to A_b\in\mathbb{R}^{4\times4}
\to \mathrm{QR}(A_b)\to R_b'\to A_b'\to \mathrm{IDCT}\to\text{RGB}.
\]

Vì vậy đây là một phương pháp QR trong **miền biến đổi DCT**.

## 2. Ma trận QR

Với mỗi block luminance \(Y_b\in\mathbb R^{8\times8}\), tính

\[
C_b=\operatorname{DCT}_2(Y_b).
\]

Lấy vùng tần số thấp 4×4:

\[
A_b=C_b[0:4,0:4].
\]

Để không mất block khi \(\det(A_b)=0\) và vẫn đủ đúng 4096 block cho watermark
64×64, dùng regularization cố định chỉ trong bước QR:

\[
\widetilde A_b=A_b+\lambda I_4,\qquad \lambda=1.
\]

Sau đó phân rã QR chuẩn hóa dấu:

\[
\widetilde A_b=Q_bR_b,
\]

với đường chéo của \(R_b\) không âm.

## 3. Vị trí nhúng trên R

Tài liệu gốc gợi ý rằng \(R_{11}\) tập trung năng lượng lớn và nhiều lược đồ
nhúng trên hàng đầu của \(R\). Proposal mới giữ \(R_{11}\) không đổi để làm
energy anchor và dùng cặp phần tử cùng hàng:

\[
v_b=\frac{R_{12}-R_{13}}{2}.
\]

Điểm mạnh của lựa chọn này là thay vì lượng tử trực tiếp \(R_{11}\), ta thay đổi
hai hệ số theo hướng đối xứng nên giữ nguyên

\[
R_{12}+R_{13}.
\]

## 4. Luật nhúng parity-QIM

Với bit \(u_b\in\{0,1\}\), bước lượng tử \(\Delta=8\), chọn số nguyên gần nhất
có parity bằng bit:

\[
k_b^*=\arg\min_{k\in\mathbb Z,\;k\bmod2=u_b}|k\Delta-v_b|,
\qquad
v_b^*=k_b^*\Delta.
\]

Đặt

\[
\delta_b=v_b^*-v_b.
\]

Cập nhật trực tiếp trên ma trận R:

\[
R_{12}'=R_{12}+\delta_b,
\qquad
R_{13}'=R_{13}-\delta_b.
\]

Khi đó

\[
\frac{R_{12}'-R_{13}'}{2}=v_b^*.
\]

Không thay đổi \(R_{11}\).

## 5. Tái tạo ảnh

Tái tạo ma trận DCT 4×4:

\[
A_b'=Q_bR_b'-\lambda I_4.
\]

Thay vùng 4×4 tương ứng trong \(C_b\), thực hiện IDCT, sau đó phân bố biến đổi
luminance về RGB bằng nghiệm hiệu chỉnh L2 nhỏ nhất. Hai closure rounds được dùng
để sửa các lỗi parity nhỏ do làm tròn `uint8` và clipping.

## 6. Giải mã

Không cần ảnh gốc. Từ ảnh nhận được, lặp lại DCT và QR trên
\(\widetilde A_b\), tính

\[
\widehat v_b=\frac{\widehat R_{12}-\widehat R_{13}}{2},
\]

và giải mã

\[
\widehat u_b=\operatorname{round}(\widehat v_b/\Delta)\bmod2.
\]

Seed chỉ xác định permutation block; key không lưu hệ số của ảnh gốc.

## 7. Trạng thái thực nghiệm hiện tại

Đây là **proposal mới, smoke-validated**, chưa phải kết quả publication-grade.
Kiểm tra nhỏ hiện tại dùng Lenna 512×512, watermark 64×64, seed 2026.
Kết quả máy đọc được nằm tại `results/dct_qr_direct_r_small_attack.json`.

Các attack hình học mạnh chưa có synchronization riêng, vì vậy không nên tuyên
bố robustness tổng quát trước khi chạy benchmark nhiều host/watermark/seed.
