# Proposal mới: DCT-QR Direct-R11 QIM

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

## 1. Mục tiêu

Phương pháp này **không phân rã QR trực tiếp trên khối ảnh miền không gian**. QR chỉ được thực hiện sau khi ảnh đã được chuyển sang miền DCT.

Pipeline:

\[
I \rightarrow Y \rightarrow \mathrm{DCT}_{8\times 8}
\rightarrow A\in\mathbb{R}^{4\times4}
\rightarrow A_\lambda=A+\lambda I
\rightarrow A_\lambda=QR
\rightarrow R_{11}'
\rightarrow QR'
\rightarrow \mathrm{IDCT}.
\]

## 2. Carrier R11

Với QR chuẩn hoá để các phần tử đường chéo của \(R\) dương,

\[
R_{11}=\|A_\lambda[:,1]\|_2.
\]

Do \(A_\lambda\) được lấy từ vùng low-frequency của DCT, \(R_{11}\) đại diện cho năng lượng mạnh của cột low-frequency đầu tiên. Proposal này dùng chính \(R_{11}\) làm carrier.

## 3. Luật nhúng

Cho bước lượng tử \(\Delta\) và bit \(b\in\{0,1\}\):

\[
k=\operatorname{round}(R_{11}/\Delta).
\]

Chọn chỉ số gần nhất \(k^*\) sao cho

\[
k^*\bmod 2=b,
\]

rồi đặt

\[
R'_{11}=k^*\Delta.
\]

Các phần tử khác của \(R\) không bị chỉnh trực tiếp. Sau đó dựng lại

\[
A_\lambda'=QR',\qquad A'=A_\lambda'-\lambda I,
\]

đưa \(A'\) trở lại block DCT và thực hiện IDCT.

## 4. Tách watermark

Từ ảnh cần kiểm tra, thực hiện lại DCT và QR, lấy \(\widehat R_{11}\):

\[
\widehat k=\operatorname{round}(\widehat R_{11}/\Delta),
\qquad
\widehat b=\widehat k\bmod 2.
\]

Không cần ảnh gốc khi extraction.

## 5. Cấu hình mặc định

- DCT block: 8x8.
- QR matrix: 4x4 low-frequency DCT.
- Watermark: 64x64 = 4096 bit.
- `step = 12.0`.
- `regularization = 1.0`.
- `closure_rounds = 2`.
- `seed = 2026`.

`closure_rounds` được dùng để sửa lại một số parity có thể thay đổi sau khi ma trận được đưa về ảnh uint8.

## 6. Khác với proposal Direct-R trước

- `dct_qr_direct_r`: giữ nguyên \(R_{11}\), nhúng trên \((R_{12}-R_{13})/2\).
- `dct_qr_r11_qim`: **thay đổi trực tiếp \(R_{11}\)** bằng parity-QIM.

Cả hai đều dùng QR trong **miền biến đổi DCT**, không phải miền không gian.

## 7. Smoke test thực tế trên Lenna

Với cấu hình mặc định `step=12`, `regularization=1`, `closure_rounds=2`:

- Embedding PSNR: khoảng **50.03 dB**.
- Embedding SSIM: khoảng **0.99792**.
- Clean: **NC = 1.0, BER = 0**.
- JPEG Q90: BER khoảng **0.10%**.
- Gaussian noise, sigma=1: **BER = 0**.
- Resize 0.9: BER khoảng **0.29%**.
- Gaussian blur radius 0.5: BER khoảng **2.25%**.
- Brightness x0.95: BER khoảng **49%**.

Kết quả brightness là một giới hạn quan trọng của việc lượng tử trực tiếp giá trị tuyệt đối `R11`: phép scale cường độ làm dịch chuyển `R11` qua nhiều ô lượng tử. Vì vậy proposal này nên được mô tả là một **R11 direct-QIM baseline/proposal mới**, chưa nên claim bất biến với biến đổi quangometric.

## 8. Clean check trên 13 host

Tất cả 13 ảnh đều giữ determinant của ma trận QR regularized khác 0. PSNR thấp nhất trong phép kiểm tra khoảng **49.92 dB**. 11/13 host cho BER clean bằng 0 với cấu hình mặc định; `milkdrop.bmp` có BER khoảng 0.24% và `tiffany.bmp` khoảng 0.49%. Đây chủ yếu là ảnh hưởng của việc tái dựng về lattice RGB uint8/clipping đối với carrier `R11` tuyệt đối.
