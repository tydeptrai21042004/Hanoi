# Báo cáo khoa học Spatial CD-DetQR

## 1. Sóng mang

Với mỗi phân hoạch Hadamard cân bằng của khối ảnh, xây dựng

\[
A_{c,p}=\begin{bmatrix}\mu_c(S_+)&1\\\mu_c(S_-)&1\end{bmatrix}
=Q_{c,p}R_{c,p}.
\]

Định thức có dấu

\[
d_{c,p}=\det(Q_{c,p})r_{11}r_{22}
=\mu_c(S_+)-\mu_c(S_-)
\]

được dùng để tạo sóng mang sai khác kênh

\[
x=d_{c_1,p}-d_{c_2,p}.
\]

## 2. Quy luật nhúng dạng đóng

Cập nhật phản đối xứng \(+saP\) và \(-saP\) trên hai kênh cho

\[
x'=x+4sa.
\]

Do đó biên độ nguyên nhỏ nhất thỏa biên dấu là

\[
a^*=\max\left(0,\left\lceil\frac{m-sx}{4}\right\rceil\right).
\]

Đây là lý do toán học của PSNR cao: mỗi bit chỉ dùng mức thay đổi nhỏ nhất cần thiết.

## 3. Pilot và đồng bộ

Phiên bản hiện tại dùng 71 pilot. Với phép biến đổi ứng viên \(T\), điểm pilot là

\[
S(T)=\frac1{N_p}\sum_i\tanh\left(
\frac{x_i(T)}{\operatorname{median}_j|x_j(T)|+\varepsilon}
\right).
\]

Chỉ chấp nhận \(T\) khi

\[
S(T)-S(I)>0.30,\qquad S(T)\geq0.50.
\]

Hai điều kiện lần lượt kiểm soát mức cải thiện so với ảnh không hiệu chỉnh và chất lượng tuyệt đối của bằng chứng pilot.

## 4. Kết quả

Trên 13 ảnh:

- PSNR trung bình: 53.454653 → **53.636001 dB**.
- NC trung bình quay/shear: 0.546462 → **0.986670**.
- NC trung bình 17 tấn công: 0.924018 → **0.975800**.
- NC sạch: **1.000000**.

Chi tiết phương pháp và chứng minh thành phần nằm trong `docs/SCIENTIFIC_METHODS_VI.md`.
