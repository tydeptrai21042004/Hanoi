# Trình bày khoa học của ba phương pháp đề xuất

> **Archived historical document.** This file is retained to preserve the original three-proposal snapshot. The live repository as of 4 September 2026 exposes eight proposals, including the three new non-DCT methods `spatial_qr`, `spatial_qr_direct_r`, and `spatial_qr_r11_qim`. The historical body below is intentionally not rewritten.

Tài liệu này chỉ giữ các thành phần có vai trò toán học rõ ràng. Mỗi thành phần phải trả lời được: đại lượng nào được xây dựng, vì sao đại lượng đó phù hợp, quy tắc quyết định là gì và tiêu chuẩn nào dùng để chấp nhận kết quả.

## 1. DCT-QR điều chỉnh cường độ theo độ ổn định

### Câu hỏi nghiên cứu

Có thể dùng độ ổn định của ma trận QR để phân phối cường độ nhúng, sao cho khối yếu được bảo vệ mạnh hơn và khối ổn định gây méo ít hơn hay không?

### Trường quan sát

\[
F=0.299R+0.587G+0.114B+0.07\left(R-\frac{G+B}{2}\right).
\]

Phần độ chói giữ cấu trúc chính của ảnh; thành phần đối kháng màu bổ sung một hướng biến thiên khác độ chói nhưng vẫn nhỏ và có thể giải thích.

### Sóng mang DCT

\[
u=C(0,1)-C(1,0).
\]

Hiệu hai hệ số AC tần số thấp làm giảm phụ thuộc vào độ sáng DC. Do cơ sở DCT trực chuẩn, độ dịch hệ số liên hệ trực tiếp với năng lượng nhiễu trong miền ảnh.

### Hai lớp QIM

\[
\mathcal Q_b(\Delta)=\{(2k+b)\Delta:k\in\mathbb Z\}.
\]

Giá trị mới là điểm gần nhất thuộc lớp của bit cần nhúng. Đây là nghiệm méo nhỏ nhất dưới ràng buộc phân tách hai lớp.

### Chứng chỉ QR

Với ma trận phân tích \(A_b=Q_bR_b\), độ tin cậy được xây dựng từ định thức, độ cân bằng đường chéo và mức liên kết ngoài đường chéo của \(R_b\). Ma trận gần suy biến có độ nhạy lớn hơn trước nhiễu, vì vậy cần biên QIM lớn hơn.

### Quy luật phân phối cường độ

\[
\Delta_b=
\begin{cases}
1.28125\Delta,&20\%\text{ khối yếu nhất},\\
\Delta,&40\%\text{ khối trung gian},\\
0.8125\Delta,&40\%\text{ khối ổn định nhất}.
\end{cases}
\]

Quy luật này là xấp xỉ rời rạc của bài toán tối thiểu hóa tổng méo với ràng buộc xác suất lỗi.

### Kết quả đã kiểm chứng trên 13 ảnh, 15 phép tấn công

- PSNR trung bình: 45.635875 → **45.684599 dB**.
- NC trung bình: 0.990520 → **0.990990**.
- NC trường hợp xấu trung bình: 0.959743 → **0.962965**.
- NC sạch: **1.000000**.

Đóng góp mới hợp lý là việc QR trực tiếp điều khiển cường độ QIM, chứ không phải chỉ ghép DCT, QIM và QR.

## 2. Giả thuyết cứu hộ DCT-Schur

### Trạng thái khoa học

Đây là giả thuyết đang khảo sát, chưa phải cải tiến đã được xác nhận. Kênh Schur phải đạt

\[
\operatorname{Accuracy}_{\text{Schur,sạch}}\geq0.99
\]

trước khi được dùng để khẳng định khả năng cứu hộ.

### Phân rã Schur thực

\[
A=ZTZ^T,\qquad T=D+N,
\]

trong đó \(D\) chứa các khối đường chéo \(1\times1\) và \(2\times2\), còn \(N\) là phần khối tam giác trên nghiêm ngặt.

### Đại lượng không chuẩn tắc

\[
\nu(A)=
\frac{\sqrt{\|A\|_F^2-\sum_i|\lambda_i(A)|^2}}
{\sqrt{\sum_i|\lambda_i(A)|^2}+1}.
\]

Thay đổi

\[
T'=D+\alpha N
\]

giữ nguyên các khối đường chéo, nên trong số học chính xác giữ nguyên phổ, vết và định thức; đồng thời \(\nu(A)\) thay đổi và có thể mang bit phụ.

### Kết quả kiểm chứng

- PSNR tăng 0.094254 dB.
- NC trung bình giảm 0.000048.
- Độ chính xác sạch của phần Schur được chọn chỉ đạt 0.858173, thấp hơn chuẩn 0.99.

Vì vậy biến thể đã thử bị loại. Hướng tiếp theo phải giải bài toán chiếu có ràng buộc và đặt biến thiên Schur trong không gian trực giao với gradient của sóng mang DCT chính.

## 3. Spatial CD-DetQR với đồng bộ định thức

### Phân hoạch không gian

Mỗi khối \(8\times8\) được biểu diễn bằng trung bình của các ô và chia thành hai tập cân bằng \(S_+,S_-\) theo mẫu Hadamard.

### Định thức QR có dấu

\[
A_{c,p}=
\begin{bmatrix}
\mu_c(S_+) & 1\\
\mu_c(S_-) & 1
\end{bmatrix}
=Q_{c,p}R_{c,p},
\]

\[
d_{c,p}=\det(Q_{c,p})r_{11}r_{22}
=\mu_c(S_+)-\mu_c(S_-).
\]

### Sóng mang sai khác kênh

\[
x=d_{c_1,p}-d_{c_2,p}.
\]

Sai khác kênh làm giảm ảnh hưởng của biến đổi chung lên các kênh màu.

### Cập nhật phản đối xứng chính xác

Thêm \(+saP\) vào kênh thứ nhất và \(-saP\) vào kênh thứ hai. Khi đó

\[
x'=x+4sa.
\]

Biên độ nguyên nhỏ nhất thỏa biên quyết định là

\[
a^*=\max\left(0,
\left\lceil\frac{m-sx}{4}\right\rceil
\right).
\]

Đây là nghiệm dạng đóng cho bài toán méo nhỏ nhất dưới ràng buộc dấu, giải thích PSNR cao mà không cần cơ chế sửa lặp.

### Đồng bộ bằng pilot cùng miền định thức

Với phép biến đổi ứng viên \(T\), điểm pilot là

\[
S(T)=\frac1{N_p}\sum_i
\tanh\left(
\frac{x_i(T)}{\operatorname{median}_j|x_j(T)|+\varepsilon}
\right).
\]

Chỉ chấp nhận phép biến đổi khi

\[
S(T^*)-S(I)>0.30,
\qquad
S(T^*)\geq0.50.
\]

Điều kiện thứ nhất so sánh với giả thuyết không biến đổi; điều kiện thứ hai ngăn ảnh không chứa watermark được chấp nhận do tăng điểm ngẫu nhiên yếu.

### Kết quả đã kiểm chứng

- PSNR trung bình: 53.454653 → **53.636001 dB**.
- NC trung bình dưới quay và shear: 0.546462 → **0.986670**.
- NC trung bình trên 17 phép tấn công: 0.924018 → **0.975800**.
- NC sạch: **1.000000**.

Đóng góp mới hợp lý là sự thống nhất giữa sóng mang định thức, quy luật cập nhật chính xác và tiêu chuẩn đồng bộ trong cùng một miền toán học.

## Tiêu chuẩn chấp nhận chung

1. NC sạch phải bằng 1.
2. Cải tiến phải tăng đồng thời ít nhất một chỉ số chất lượng và một chỉ số bền vững, không làm giảm đáng kể chỉ số tổng hợp quan trọng.
3. Kênh cứu hộ phải được kiểm tra độc lập trước khi hợp nhất.
4. Đồng bộ chỉ được dùng pilot, không được dùng ảnh gốc hoặc watermark gốc.
5. Kết quả chi tiết nằm trong `results/scientific_validation/`.
