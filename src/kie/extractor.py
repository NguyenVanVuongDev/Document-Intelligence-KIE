import re
import math
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz
from src.kie.spatial import SpatialBox, LogicalLine, build_logical_lines

def strip_vi_accents(s: str) -> str:
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


RECEIPT_TITLES_BLACKLIST = (
    "HÓA ĐƠN", "HOA DON", "PHIẾU THANH TOÁN", "PHIEU THANH TOAN",
    "PHIẾU TÍNH TIỀN", "PHIEU TINH TIEN", "PHIẾU THU", "PHIEU THU",
    "RECEIPT", "BILL", "INVOICE", "BẢNG KÊ", "TẠM TÍNH", "BÁN HÀNG"
)

NON_ADDRESS_KEYWORDS = (
    "HÓA ĐƠN", "HOA DON", "PHIẾU", "RECEIPT", "ĐIỆN THOẠI", "DIEN THOAI",
    "ĐT:", "DT:", "TEL", "HOTLINE", "WEBSITE", "WWW.", "MST:", "MÃ SỐ THUẾ",
    "THU NGÂN", "THU NGAN", "NHÂN VIÊN", "NHAN VIEN", "KHÁCH HÀNG",
    "NGÀY:", "NGAY:", "GIỜ:", "GIO:", "CA:", "SỐ HĐ", "SO HD",
    "SIÊU THỊ", "SIEU THI", "WINMART+", "VINMART+", "CÔNG TY", "CONG TY"
)

def clean_ocr_date_noise(text: str) -> str:
    """Sửa lỗi OCR hay nhầm chữ O->0, l/I->1 xung quanh dấu / hoặc - của ngày tháng."""
    s = text
    s = re.sub(r"(?<=\d)[Oo](?=[/\-.\d])", "0", s)
    s = re.sub(r"(?<=[/\-.])[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[Il](?=[/\-.\d])", "1", s)
    s = re.sub(r"(?<=[/\-.])[Il](?=\d)", "1", s)
    return s

def normalize_date(val: str) -> str:
    if not val:
        return ""
    s = clean_ocr_date_noise(str(val).strip())
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(20\d{2}|19\d{2}|\d{2})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= d <= 31 and 1 <= mo <= 12:
            if len(y) == 2:
                y = "20" + y if int(y) < 50 else "19" + y
            return f"{d:02d}/{mo:02d}/{y}"
    return ""

def extract_money_candidates(text: str, normalize: bool = True) -> List[str]:
    """
    Trích xuất các chuỗi số tiền hợp lệ trên hóa đơn Việt Nam.
    Hỗ trợ cả trường hợp OCR chèn khoảng trắng quanh dấu chấm/phẩy phân cách hàng nghìn
    (ví dụ: '398. 125', '164, 435', '70 000', '150.000đ').
    """
    if not text:
        return []
    cleaned = re.sub(r"(?<=[\d.,])[Oo](?=[\d.,\sđdVNDvnd]|$)", "0", text)
    cleaned = re.sub(r"\b\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", cleaned)

    pattern = r"(?<!\d)(\d{1,3}(?:\s*[.,]\s*\d{3})+(?:\s*[.,]\s*\d{1,2})?|\d{1,3}(?:\s\d{3})+|\d{4,9})(?:\s*(?:đ|d|vnd|vnđ))?(?!\d)"
    matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
    results = []
    for m in matches:
        raw = m.strip()
        raw = re.sub(r"\s*[.,]\s*00$", "", raw)
        digits = re.sub(r"[^\d]", "", raw)
        if not digits or not digits.isdigit():
            continue
        if digits.startswith("0") and len(digits) >= 9:
            continue
        val_int = int(digits)
        if 1000 <= val_int <= 500_000_000:
            results.append(digits if normalize else raw)
    return results

def _fuzzy_has_keyword(text: str, keywords: List[str], cutoff: int = 76) -> Tuple[bool, float]:
    """Kiểm tra chuỗi text có khớp mờ (Levenshtein / RapidFuzz) với bất kỳ từ khóa nào không."""
    if not text:
        return False, 0.0
    t_up = text.upper()
    t_norm = strip_vi_accents(t_up).replace("0", "O")
    best = 0.0
    for kw in keywords:
        kw_up = kw.upper()
        kw_ascii = strip_vi_accents(kw_up)
        if kw_up in t_up or kw_ascii in t_norm:
            return True, 100.0
        if len(t_norm) >= max(len(kw_ascii) - 2, 4):
            score = max(
                fuzz.partial_ratio(kw_up, t_up),
                fuzz.partial_ratio(kw_ascii, t_norm),
            )
            if score > best:
                best = score
    return best >= cutoff, best

# ==============================================================================
# 1. BASELINE KIE (Raw OCR Order + Pure Regex, KHÔNG dùng Spatial Clustering)
# ==============================================================================
def extract_baseline_regex(raw_boxes: List[Any], cfg: dict) -> Dict[str, Any]:
    texts = [str(getattr(b, "text", None) or b.get("text", "")).strip() for b in raw_boxes]
    texts = [t for t in texts if t]
    full_text = "\n".join(texts)

    out = {"merchant": "", "date": "", "total": "", "address": ""}
    if not texts:
        return {"fields": out, "full_text": ""}

    out["merchant"] = texts[0]

    m_date = re.search(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", full_text)
    if m_date:
        out["date"] = normalize_date(m_date.group(1))

    m_addr = re.search(r"(?:địa chỉ|đ/c|address)[:\s]*([^\n]{5,80})", full_text, flags=re.IGNORECASE)
    if m_addr:
        out["address"] = m_addr.group(1).strip()

    m_tot = re.search(r"(?:tổng cộng|tổng tiền|thanh toán|total)[^\d]{0,15}(\d{1,3}(?:[.,]\d{3})+)", full_text, flags=re.IGNORECASE)
    if m_tot:
        cands = extract_money_candidates(m_tot.group(1), normalize=True)
        if cands:
            out["total"] = cands[0]

    return {"fields": out, "full_text": full_text}


# ==============================================================================
# 2. FULL SPATIAL KIE (Y-Clustering + X-Sorting + Fuzzy Anchor + Spatial Lookup)
# ==============================================================================
def extract_spatial_kie(raw_boxes: List[Any], cfg: dict,
                        img_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    kie_cfg = cfg.get("kie", {})
    sp_cfg  = kie_cfg.get("spatial", {})
    fz_cfg  = kie_cfg.get("fuzzy", {})
    anc_cfg = kie_cfg.get("anchors", {})

    y_overlap = float(sp_cfg.get("y_overlap_ratio", 0.50))
    lines = build_logical_lines(raw_boxes, y_overlap_ratio=y_overlap)
    if not lines:
        return {
            "fields": {"merchant": "", "date": "", "total": "", "address": ""},
            "lines": [],
            "full_text": "",
        }

    min_x = min(ln.xmin for ln in lines)
    max_x = max(ln.xmax for ln in lines)
    min_y = min(ln.ymin for ln in lines)
    max_y = max(ln.ymax for ln in lines)
    W = float(img_shape[1]) if img_shape is not None and len(img_shape) >= 2 else max(max_x + min_x, 1.0)
    H = float(img_shape[0]) if img_shape is not None and len(img_shape) >= 2 else max(max_y + min_y, 1.0)
    receipt_center_x = (min_x + max_x) / 2.0

    # ---- A. Trích xuất MERCHANT NAME ----
    merchant_hints = anc_cfg.get("merchant_hints", [])
    addr_keywords = anc_cfg.get("address", [])
    top_k = lines[:min(5, len(lines))]
    max_h_top = max((ln.max_height for ln in top_k), default=1.0)

    best_merchant = ""
    best_m_idx = -1
    best_m_score = -1e9
    for idx, ln in enumerate(top_k):
        txt = ln.text.strip()
        if len(txt) < 3:
            continue
        txt_up = txt.upper()
        if any(blk in txt_up for blk in RECEIPT_TITLES_BLACKLIST) and len(txt) < 28:
            continue
        if re.fullmatch(r"[\d\s/\-:.,]+", txt):
            continue
        if re.match(r"^(?:địa chỉ|dia chi|đ/c|đc:|dc:|tổ\s+\d|khu\s+|số\s+\d|tp\.?\s|vm\+)", txt_up, flags=re.IGNORECASE):
            continue

        h_score = ln.max_height / max(max_h_top, 1.0)
        center_dist = abs(ln.cx - receipt_center_x) / max(W * 0.5, 1.0)
        c_score = max(0.0, 1.0 - center_dist)
        pos_score = 1.0 if idx == 0 else (0.85 if idx <= 2 else 0.4)
        has_hint, _ = _fuzzy_has_keyword(txt, merchant_hints, cutoff=82)
        hint_bonus = 0.9 if has_hint else 0.0

        total_score = 1.4 * h_score + 0.7 * c_score + 0.7 * pos_score + hint_bonus
        if total_score > best_m_score:
            best_m_score = total_score
            best_merchant = txt
            best_m_idx = idx

    #Làm sạch hậu tố tỉnh/thành nếu vô tình dính vào merchant
    best_merchant = re.sub(r"\s+(?:TP\.?\s*Cẩm Phả|T\.?\s*Quảng Ninh|Quảng Ninh).*$", "", best_merchant, flags=re.IGNORECASE).strip(" ,.-") or best_merchant

    # ---- B. Trích xuất ADDRESS (Gom tối đa 3 dòng địa chỉ liên tiếp ở nửa trên hóa đơn) ----
    addr_cutoff = int(fz_cfg.get("address_score_cutoff", 78))
    extracted_address = ""
    top_half_limit = max(int(len(lines) * 0.50), min(9, len(lines)))
    addr_lines_collected = []

    def _is_addr_line(cand_txt: str) -> bool:
        if len(cand_txt) < 4:
            return False
        c_up = cand_txt.upper()
        if any(bad in c_up for bad in NON_ADDRESS_KEYWORDS):
            if not re.search(r"(?:địa chỉ|dia chi|đ/c|đc:|dc:|vm\+)", c_up, flags=re.IGNORECASE):
                return False
        m_a, _ = _fuzzy_has_keyword(cand_txt, addr_keywords, cutoff=addr_cutoff)
        has_p = bool(re.search(
            r"(?:đ/c|địa chỉ|tổ\s*\d|khu\s*\w|số\s*\d|đường|phố|phường|p\.|quận|q\.|huyện|tỉnh|tp\.|thành phố|cẩm phả|quảng ninh|hà nội|vm\+|qnh)",
            cand_txt, flags=re.IGNORECASE
        ))
        return m_a or has_p

    for idx in range(top_half_limit):
        if idx == best_m_idx:
            continue
        ln_txt = lines[idx].text.strip()
        if _is_addr_line(ln_txt):
            addr_lines_collected.append(ln_txt)
            for nxt_idx in range(idx + 1, min(idx + 4, top_half_limit)):
                if nxt_idx == best_m_idx:
                    continue
                nxt_txt = lines[nxt_idx].text.strip()
                if _is_addr_line(nxt_txt):
                    addr_lines_collected.append(nxt_txt)
                elif len(addr_lines_collected) >= 2:
                    break
            break

    if addr_lines_collected:
        extracted_address = " ".join(addr_lines_collected)

    # ---- C. Trích xuất DATE & TIME ----
    extracted_date = ""
    for ln in lines:
        d_norm = normalize_date(ln.text)
        if d_norm:
            extracted_date = d_norm
            break
    if not extracted_date:
        full_joined = " ".join(ln.text for ln in lines)
        extracted_date = normalize_date(full_joined)

    # ---- D. Trích xuất TOTAL AMOUNT (Spatial + Fuzzy Anchor + Bottom-Half Fallback) ----
    total_anchors = anc_cfg.get("total", [])
    exclude_anchors = anc_cfg.get("exclude_total", [])
    total_cutoff = int(fz_cfg.get("total_score_cutoff", 76))
    x_tol_ratio = float(sp_cfg.get("same_col_x_tol_ratio", 0.35))

    anchor_candidates: List[Tuple[float, int, str]] = []

    for idx, ln in enumerate(lines):
        ln_txt = ln.text
        ln_up = ln_txt.upper()
        is_excluded, _ = _fuzzy_has_keyword(ln_txt, exclude_anchors, cutoff=82)
        if is_excluded and not any(k in ln_up for k in ("TỔNG CỘNG", "TỔNG THANH TOÁN", "PHẢI T.TOÁN", "PHẢI THU", "TỔNG TIỀN")):
            continue

        anchor_box: Optional[SpatialBox] = None
        best_fz = 0.0
        for b in ln.boxes:
            ok, sc = _fuzzy_has_keyword(b.text, total_anchors, cutoff=total_cutoff)
            if ok and sc > best_fz:
                best_fz = sc
                anchor_box = b

        if anchor_box is None:
            ok_ln, sc_ln = _fuzzy_has_keyword(ln_txt, total_anchors, cutoff=total_cutoff)
            if ok_ln:
                best_fz = sc_ln
                anchor_box = ln.boxes[0]

        if anchor_box is not None:
            rel_y = idx / max(len(lines) - 1, 1)
            pos_weight = 1.25 if rel_y >= 0.35 else 0.70

            right_texts = [b.text for b in ln.boxes if b.cx >= anchor_box.cx]
            same_line_cands = extract_money_candidates(" ".join(right_texts), normalize=True)
            if not same_line_cands:
                same_line_cands = extract_money_candidates(ln_txt, normalize=True)

            for c_str in same_line_cands:
                c_int = int(c_str)
                score = best_fz * pos_weight + 50.0 + math.log10(max(c_int, 1))
                anchor_candidates.append((score, c_int, c_str))

            if not same_line_cands:
                for offset in (1, 2):
                    if idx + offset >= len(lines):
                        continue
                    next_ln = lines[idx + offset]
                    is_next_excl, _ = _fuzzy_has_keyword(next_ln.text, exclude_anchors, cutoff=82)
                    if is_next_excl:
                        continue
                    joined_next = " ".join(
                        nb.text for nb in next_ln.boxes
                        if abs(nb.cx - anchor_box.cx) <= (W * x_tol_ratio) or nb.cx > anchor_box.cx
                    )
                    for c_str in extract_money_candidates(joined_next or next_ln.text, normalize=True):
                        c_int = int(c_str)
                        score = best_fz * pos_weight + (18.0 / offset) + math.log10(max(c_int, 1))
                        anchor_candidates.append((score, c_int, c_str))

    extracted_total = ""
    if anchor_candidates:
        anchor_candidates.sort(key=lambda x: (round(x[0], 1), x[1]), reverse=True)
        extracted_total = anchor_candidates[0][2]
    else:
        bottom_start = int(len(lines) * float(sp_cfg.get("bottom_half_threshold", 0.40)))
        bottom_cands: List[Tuple[int, str]] = []
        for ln in lines[bottom_start:]:
            is_excl, _ = _fuzzy_has_keyword(ln.text, exclude_anchors, cutoff=82)
            if is_excl:
                continue
            for c_str in extract_money_candidates(ln.text, normalize=True):
                bottom_cands.append((int(c_str), c_str))
        if bottom_cands:
            bottom_cands.sort(key=lambda x: x[0], reverse=True)
            extracted_total = bottom_cands[0][1]

    line_texts = [ln.text for ln in lines]
    return {
        "fields": {
            "merchant": best_merchant,
            "date": extracted_date,
            "total": extracted_total,
            "address": extracted_address,
        },
        "lines": line_texts,
        "full_text": "\n".join(line_texts),
    }
