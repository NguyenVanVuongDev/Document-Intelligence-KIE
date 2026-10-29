import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Tuple

@dataclass
class SpatialBox:
    """Bước 1: Chuẩn hóa tọa độ từ đa giác 4 điểm của OCR."""
    polygon: List[List[float]]
    text: str
    confidence: float
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @classmethod
    def from_ocr_box(cls, box: Any) -> "SpatialBox":
        if isinstance(box, cls):
            return box
        poly = box.polygon if hasattr(box, "polygon") else box["bbox"]
        txt  = box.text if hasattr(box, "text") else box["text"]
        conf = float(box.confidence if hasattr(box, "confidence") else box.get("conf", 1.0))
        pts  = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
        xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        w  = max(xmax - xmin, 1.0)
        h  = max(ymax - ymin, 1.0)
        return cls(
            polygon=pts.tolist(), text=str(txt).strip(), confidence=conf,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            cx=cx, cy=cy, width=w, height=h
        )

@dataclass
class LogicalLine:
    """Dòng văn bản logic sau khi gộp theo trục Y và sắp xếp theo trục X."""
    line_idx: int
    boxes: List[SpatialBox] = field(default_factory=list)

    @property
    def cy(self) -> float:
        return float(np.mean([b.cy for b in self.boxes])) if self.boxes else 0.0

    @property
    def mean_height(self) -> float:
        return float(np.mean([b.height for b in self.boxes])) if self.boxes else 1.0

    @property
    def max_height(self) -> float:
        return float(np.max([b.height for b in self.boxes])) if self.boxes else 1.0

    @property
    def xmin(self) -> float:
        return min(b.xmin for b in self.boxes) if self.boxes else 0.0

    @property
    def xmax(self) -> float:
        return max(b.xmax for b in self.boxes) if self.boxes else 0.0

    @property
    def ymin(self) -> float:
        return min(b.ymin for b in self.boxes) if self.boxes else 0.0

    @property
    def ymax(self) -> float:
        return max(b.ymax for b in self.boxes) if self.boxes else 0.0

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def text(self) -> str:
        return " ".join(b.text for b in self.boxes if b.text)

    def sort_x(self) -> None:
        """Bước 3: Sắp xếp các box trong cùng dòng từ trái sang phải theo xmin tăng dần."""
        self.boxes.sort(key=lambda b: (b.xmin, b.cx))


def is_same_line(box_a: SpatialBox, cy_b: float, h_b: float, y_overlap_ratio: float = 0.5) -> bool:
    """
    Bước 2: Kiểm tra điều kiện gộp dòng Y-Clustering:
    |cy(A) - cy(B)| < y_overlap_ratio * ((h(A) + h(B)) / 2.0)
    (Khi y_overlap_ratio = 0.5, ngưỡng chính là một nửa chiều cao trung bình của 2 box).
    """
    avg_h = 0.5 * (box_a.height + h_b)
    return abs(box_a.cy - cy_b) < (y_overlap_ratio * avg_h)


def build_logical_lines(raw_boxes: List[Any], y_overlap_ratio: float = 0.5) -> List[LogicalLine]:
    """
    Thực thi trọn vẹn 3 bước Spatial Bounding-Box Association:
      1. Chuẩn hóa từng box thành SpatialBox (cx, cy, h, xmin, xmax).
      2. Phân cụm theo dòng (Y-Clustering) dựa trên khoảng cách tâm dọc so với chiều cao trung bình.
      3. Sắp xếp trục ngang (X-Sorting) theo xmin tăng dần trong từng dòng.
    """
    sboxes = [SpatialBox.from_ocr_box(b) for b in raw_boxes if str(getattr(b, "text", None) or b.get("text", "")).strip()]
    if not sboxes:
        return []

    # Sắp xếp sơ bộ theo cy từ trên xuống dưới
    sboxes.sort(key=lambda b: (b.cy, b.xmin))

    brand_logos = ("VINCOMMERCE", "VINMART", "WINMART", "CIRCLE K", "MINISTOP", "BÁCH HÓA XANH", "CO.OP", "BIG C", "HIGHLANDS")
    addr_tokens = ("TP.", "TP ", "TỈNH", "TINH", "PHƯỜNG", "PHUONG", "P.", "QUẬN", "QUAN", "HUYỆN", "ĐƯỜNG", "DUONG", "KHU ", "TỔ ", "TO ", "CẨM PHẢ", "CAM PHA", "QUẢNG NINH", "QUANG NINH", "HÀ NỘI", "Đ/C", "ĐỊA CHỈ")

    def _is_logo_addr_collision(box_cand: SpatialBox, line_cand: LogicalLine) -> bool:
        t1 = box_cand.text.upper().strip()
        t2 = line_cand.text.upper().strip()
        b1_logo = any(k in t1 for k in brand_logos) and len(t1) <= 22
        b2_logo = any(k in t2 for k in brand_logos) and len(t2) <= 22
        b1_addr = any(k in t1 for k in addr_tokens)
        b2_addr = any(k in t2 for k in addr_tokens)
        return (b1_logo and b2_addr) or (b2_logo and b1_addr)

    lines: List[LogicalLine] = []
    for sb in sboxes:
        best_line = None
        best_dy = float("inf")
        for line in lines:
            if is_same_line(sb, line.cy, line.mean_height, y_overlap_ratio=y_overlap_ratio):
                if _is_logo_addr_collision(sb, line):
                    continue
                dy = abs(sb.cy - line.cy)
                if dy < best_dy:
                    best_dy = dy
                    best_line = line
        if best_line is not None:
            best_line.boxes.append(sb)
        else:
            lines.append(LogicalLine(line_idx=len(lines), boxes=[sb]))

    # Sắp xếp lại các dòng theo cy tăng dần và sắp xếp các box trong mỗi dòng theo xmin tăng dần
    lines.sort(key=lambda ln: ln.cy)
    for idx, ln in enumerate(lines):
        ln.line_idx = idx
        ln.sort_x()

    return lines
