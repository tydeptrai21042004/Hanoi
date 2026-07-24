#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "before_after_pso" / "comparison_summary.csv"
OUTPUT = ROOT / "docs" / "THREE_PROPOSAL_METHODS_NOVELTY_VI.pdf"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DV", FONT))
pdfmetrics.registerFont(TTFont("DV-Bold", FONT_BOLD))


def _fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(18 * mm, 12 * mm, "TÀI LIỆU ĐỀ XUẤT - BẢO MẬT")
    canvas.drawRightString(192 * mm, 12 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def main() -> None:
    if not RESULTS.exists():
        raise FileNotFoundError("Run scripts/benchmark_before_after.py first")
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    by_id = {row["method_id"]: row for row in rows}

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Ba phương pháp watermark mù và tối ưu PSO",
        author="Research proposal",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleVI", parent=styles["Title"], fontName="DV-Bold",
        fontSize=25, leading=31, alignment=TA_LEFT, spaceAfter=8 * mm,
    )
    subtitle = ParagraphStyle(
        "SubVI", parent=styles["Normal"], fontName="DV",
        fontSize=11, leading=16, textColor=colors.HexColor("#444444"),
    )
    h1 = ParagraphStyle(
        "H1VI", parent=styles["Heading1"], fontName="DV-Bold",
        fontSize=16, leading=21, spaceBefore=4 * mm, spaceAfter=3 * mm,
    )
    h2 = ParagraphStyle(
        "H2VI", parent=styles["Heading2"], fontName="DV-Bold",
        fontSize=12, leading=17, spaceBefore=3 * mm, spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "BodyVI", parent=styles["BodyText"], fontName="DV",
        fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=2.5 * mm,
    )
    small = ParagraphStyle(
        "SmallVI", parent=body, fontSize=8.5, leading=12,
        textColor=colors.HexColor("#555555"),
    )
    box = ParagraphStyle(
        "BoxVI", parent=body, borderColor=colors.HexColor("#777777"),
        borderWidth=0.7, borderPadding=8, backColor=colors.HexColor("#F2F2F2"),
        spaceBefore=3 * mm, spaceAfter=4 * mm,
    )

    story = []
    story.append(Paragraph("ĐỀ XUẤT BA PHƯƠNG PHÁP", title))
    story.append(Paragraph("BLIND WATERMARKING | DCT-QR - DCT-SCHUR RESCUE - CD-DetQR", subtitle))
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph(
        "Tài liệu này chỉ trình bày điểm mới, vai trò và kết quả ở mức tổng quan. "
        "Nội dung không phải đặc tả triển khai độc lập và không công bố carrier, "
        "công thức nội bộ, khóa, biên quyết định hoặc miền tìm kiếm tham số.", box
    ))
    story.append(Paragraph("1. Mục tiêu chung", h1))
    story.append(Paragraph(
        "Ba phương pháp cùng hướng đến watermark nhị phân 64 x 64, giải mã hoàn toàn mù, "
        "giữ chất lượng ảnh phù hợp và duy trì khả năng phục hồi qua nhóm 15 tấn công moderate. "
        "Mỗi phương pháp sử dụng một cách tạo bằng chứng khác nhau, nhưng cùng được đánh giá "
        "trên một host, một watermark và cùng tập tấn công để bảo đảm so sánh trực tiếp.", body
    ))
    story.append(Paragraph("2. Ba phương pháp được giữ lại", h1))

    methods = [
        ("DCT-QR", "DCT-QR Certified Blind Watermarking",
         "Kết hợp một carrier có độ biến dạng thấp với chứng chỉ QR nhằm đánh giá độ ổn định cục bộ. "
         "Điểm mới nằm ở việc QR không chỉ là vị trí nhúng, mà còn tham gia lựa chọn mức tin cậy khi giải mã."),
        ("DCT-Schur", "DCT-Schur Direct Rescue",
         "Giữ carrier chính ổn định và bổ sung một tín hiệu Schur yếu để hỗ trợ các quyết định chưa chắc chắn. "
         "Điểm mới là cơ chế cứu hộ có điều kiện, chỉ đóng vai trò bổ sung thay vì thay thế kênh chính."),
        ("QR đơn", "Blind CD-DetQR",
         "Hoạt động trực tiếp trong miền không gian, sử dụng quan hệ determinant QR giữa các kênh màu. "
         "Điểm mới là không cần DCT và vẫn duy trì giải mã mù với một quy tắc quyết định trực tiếp."),
    ]
    for short_name, full_name, text in methods:
        story.append(Paragraph(f"{short_name}: {full_name}", h2))
        story.append(Paragraph(text, body))

    story.append(PageBreak())
    story.append(Paragraph("3. Điểm mới ở mức công bố", h1))
    novelty_text = [
        ["Phương pháp", "Điểm mới có thể công bố ngắn"],
        ["DCT-QR", "Chứng chỉ QR được dùng như bằng chứng ổn định cho giải mã mù, không chỉ là phép phân rã phụ."],
        ["DCT-Schur Rescue", "Tín hiệu Schur phụ được dùng để hỗ trợ vùng quyết định yếu, trong khi carrier chính vẫn được giữ ổn định."],
        ["Blind CD-DetQR", "Carrier QR-determinant trong miền không gian khai thác khác biệt liên kênh và không cần biến đổi DCT."],
        ["PSO", "Tự động chọn bộ tham số cân bằng giữa chất lượng ảnh, clean NC và mean NC sau tấn công."],
    ]
    cell_style = ParagraphStyle("CellVI", parent=small, fontSize=8.2, leading=11, textColor=colors.black)
    head_style = ParagraphStyle("CellHeadVI", parent=cell_style, fontName="DV-Bold")
    novelty_data = []
    for row_index, row in enumerate(novelty_text):
        style = head_style if row_index == 0 else cell_style
        novelty_data.append([Paragraph(cell, style) for cell in row])
    table = Table(novelty_data, colWidths=[42 * mm, 128 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DV"),
        ("FONTNAME", (0, 0), (-1, 0), "DV-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.7),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("4. Tối ưu tham số bằng PSO", h1))
    story.append(Paragraph(
        "Particle Swarm Optimization được dùng như một lớp lựa chọn tham số chung. "
        "Mỗi ứng viên được đánh giá bằng chất lượng ảnh, độ chính xác khi chưa tấn công và "
        "giá trị NC trung bình trên toàn bộ 15 tấn công. Cấu hình ban đầu luôn được giữ như "
        "một hạt tham chiếu, vì vậy kết quả tối ưu không thể tự động loại bỏ baseline nếu không tìm được lựa chọn tốt hơn.", body
    ))
    story.append(Paragraph(
        "Tài liệu này không công bố số lượng hạt, miền tìm kiếm, trọng số mục tiêu hoặc bộ tham số cuối. "
        "Các thông tin đó được lưu trong mã nguồn và tệp cấu hình dành cho nhóm triển khai.", box
    ))

    story.append(PageBreak())
    story.append(Paragraph("5. Kết quả trước và sau tối ưu", h1))
    header = ["Phương pháp", "PSNR trước", "PSNR sau", "NC sạch", "NC TB trước", "NC TB sau"]
    names = {
        "dct_qr": "DCT-QR",
        "dct_schur_rescue": "DCT-Schur Rescue",
        "spatial_cd_detqr": "Blind CD-DetQR",
    }
    result_data = [header]
    for method_id in ("dct_qr", "dct_schur_rescue", "spatial_cd_detqr"):
        row = by_id[method_id]
        result_data.append([
            names[method_id],
            _fmt(row["before_psnr"]),
            _fmt(row["after_psnr"]),
            _fmt(row["after_clean_nc"]),
            _fmt(row["before_mean_nc"]),
            _fmt(row["after_mean_nc"]),
        ])
    result_table = Table(result_data, colWidths=[41 * mm, 25 * mm, 25 * mm, 21 * mm, 29 * mm, 29 * mm], repeatRows=1)
    result_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DV"),
        ("FONTNAME", (0, 0), (-1, 0), "DV-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Trên Lenna và 15 tấn công moderate, cả ba cấu hình sau PSO đều giữ NC sạch bằng 1. "
        "DCT-QR và DCT-Schur Rescue tăng rõ mean NC khi chấp nhận PSNR thấp hơn nhưng vẫn trong vùng chất lượng cao. "
        "Blind CD-DetQR cải thiện nhẹ vì cấu hình ban đầu đã ở gần biên PSNR mục tiêu.", body
    ))
    story.append(Paragraph("6. Phạm vi tuyên bố", h1))
    story.append(Paragraph(
        "Các số liệu trong tài liệu là kết quả kiểm thử trên một ảnh chủ và không được hiểu là kết luận thống kê cho mọi ảnh. "
        "Ba phương pháp vẫn là nguyên mẫu nghiên cứu. Khi công bố bên ngoài, chỉ nên mô tả điểm mới, chế độ blind và kết quả tổng quan; "
        "không nên cung cấp chi tiết đủ để tái tạo độc lập nếu chưa có thỏa thuận phù hợp.", body
    ))
    story.append(Paragraph(
        "Lưu ý: thuật ngữ 'Certified' chỉ là tên cơ chế đánh giá độ tin cậy nội bộ, không phải chứng nhận của một tổ chức độc lập.", small
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
