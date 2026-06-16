from pathlib import Path
from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'


def text_width(draw, text, font):
    return int(draw.textbbox((0, 0), str(text), font=font)[2])


def export_sheet2_image(xlsx_path, output_path, sheet_index=1, max_cols=5):
    wb = load_workbook(xlsx_path, data_only=True)
    ws = wb.worksheets[sheet_index]

    # Capture columns A:E from row 1 through the last row that has data in A:E.
    last_row = 1
    for r in range(1, ws.max_row + 1):
        if any(ws.cell(r, c).value not in (None, '') for c in range(1, max_cols + 1)):
            last_row = r

    data = []
    for r in range(1, last_row + 1):
        row = []
        for c in range(1, max_cols + 1):
            v = ws.cell(r, c).value
            if v is None:
                v = ''
            elif isinstance(v, float):
                v = f'{v:.1f}'.rstrip('0').rstrip('.')
            else:
                # GitHub runners may not have emoji-capable fonts; use a plain marker.
                v = str(v).replace('✅', '有')
            row.append(v)
        data.append(row)

    font = ImageFont.truetype(FONT_PATH, 22)
    header_font = ImageFont.truetype(FONT_PATH, 22)
    tmp_img = Image.new('RGB', (1, 1), 'white')
    draw = ImageDraw.Draw(tmp_img)

    # Auto-fit columns with padding. Keep it compact like the Excel crop.
    col_widths = []
    for c in range(max_cols):
        widest = max(text_width(draw, row[c], header_font if i == 0 else font) for i, row in enumerate(data))
        col_widths.append(max(92, widest + 28))

    row_h = 34
    margin = 12
    title_h = 42
    w = sum(col_widths) + margin * 2
    h = title_h + row_h * len(data) + margin * 2

    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((margin, margin - 2), f'{Path(xlsx_path).stem}｜{ws.title}', fill=(40, 40, 40), font=header_font)

    y0 = margin + title_h
    grid = (205, 205, 205)
    header_bg = (238, 242, 247)

    for r, row in enumerate(data):
        y = y0 + r * row_h
        x = margin
        for c, val in enumerate(row):
            if r == 0:
                draw.rectangle([x, y, x + col_widths[c], y + row_h], fill=header_bg)
            draw.rectangle([x, y, x + col_widths[c], y + row_h], outline=grid)
            draw.text((x + 8, y + 4), val, fill=(20, 20, 20), font=header_font if r == 0 else font)
            x += col_widths[c]

    img.save(output_path)
    return output_path


if __name__ == '__main__':
    import sys
    export_sheet2_image(sys.argv[1], sys.argv[2])
