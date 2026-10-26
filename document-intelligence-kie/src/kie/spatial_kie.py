import re
import numpy as np
from typing import List, Dict, Any

def box_bounds(poly: List[List[float]]):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return x0, y0, x1, y1, (x0 + x1) / 2.0, (y0 + y1) / 2.0, max(y1 - y0, 1.0)

def cluster_text_lines(boxes: List[Any], y_overlap_ratio: float = 0.5) -> List[List[Any]]:
    """Gom các bounding box có tọa độ Y giao nhau vào cùng một dòng ngang và sắp xếp từ trái sang phải."""
    if not boxes:
        return []
    annotated = []
    for b in boxes:
        poly = b.polygon if hasattr(b, "polygon") else b["bbox"]
        annotated.append((b, box_bounds(poly)))
    annotated.sort(key=lambda item: item[1][5])  # sort theo y_center

    lines = []
    for b, bounds in annotated:
        _, y0, _, y1, _, yc, h = bounds
        placed = False
        for line in lines:
            _, ly0, _, ly1, _, lyc, lh = line[0][1]
            if abs(yc - lyc) <= max(h, lh) * y_overlap_ratio:
                line.append((b, bounds))
                placed = True
                break
        if not placed:
            lines.append([(b, bounds)])

    sorted_lines = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])  # sort trái sang phải theo x0
        sorted_lines.append([item[0] for item in line])
    return sorted_lines

def extract_entities_spatial(boxes: List[Any], cfg: dict) -> Dict[str, Any]:
    """Gom dòng không gian (Spatial Clustering) kết hợp Regex để trích xuất Key-Value cho Day 2."""
    kie_cfg = cfg.get("kie", {})
    sc_cfg = kie_cfg.get("spatial_clustering", {})
    y_ratio = sc_cfg.get("line_y_overlap_ratio", 0.5)

    lines = cluster_text_lines(boxes, y_overlap_ratio=y_ratio)
    line_texts = [" ".join((b.text if hasattr(b, "text") else b["text"]) for b in ln) for ln in lines]
    full_text = "
".join(line_texts)

    patterns = kie_cfg.get("patterns", {})
    extracted = {"vendor": "", "date": "", "invoice_id": "", "total_amount": ""}

    # 1. Vendor thường nằm ở 1-3 dòng đầu tiên của hóa đơn
    for ln in line_texts[:3]:
        if len(ln.strip()) >= 4 and not re.search(r"^(hóa đơn|phiếu|receipt|bill)", ln.strip(), re.I):
            extracted["vendor"] = ln.strip()
            break

    # 2. Quét từng dòng đã gom cụm không gian (Spatial Line) để bắt cặp Key-Value cùng dòng hoặc dòng kế tiếp
    for idx, ln in enumerate(line_texts):
        context = ln
        if idx + 1 < len(line_texts):
            context_Below = ln + " " + line_texts[idx + 1]
        else:
            context_Below = ln

        for field_name in ("total_amount", "date", "invoice_id"):
            pat = patterns.get(field_name)
            if not pat or extracted.get(field_name):
                continue
            m = re.search(pat, context, flags=re.IGNORECASE) or re.search(pat, context_Below, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip() if m.lastindex else m.group(0).strip()
                if field_name == "total_amount" and kie_cfg.get("normalize_amount", True):
                    digits = re.sub(r"[^\d]", "", val)
                    val = digits if digits else val
                extracted[field_name] = val

    return {"fields": extracted, "lines": line_texts, "full_text": full_text}
