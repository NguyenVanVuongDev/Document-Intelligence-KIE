import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

def safe_imread(path: Path) -> np.ndarray:
    """Đọc ảnh an toàn bằng cv2.imdecode; ném IOError nếu ảnh hỏng, tuyệt đối không thay bằng ảnh trắng."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"Empty file: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2.imdecode failed: {path}")
    return img

def remove_shadow(img: np.ndarray, bg_kernel_size: int = 51,
                  median_blur_ksize: int = 51, mode: str = "divide") -> np.ndarray:
    if img is None:
        raise ValueError("remove_shadow: image is None")
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
    else:
        L, A, B = img.copy(), None, None
    ksize = max(int(bg_kernel_size) | 1, 31)
    msize = max(int(median_blur_ksize) | 1, 31)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k)
    bg = cv2.medianBlur(bg, msize)
    bg_f = np.clip(bg, 1, 255).astype(np.float32)
    Lf = L.astype(np.float32)
    flat = cv2.divide(Lf, bg_f, scale=255.0) if mode == "divide" else (Lf - bg_f + 128.0)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    bg_std = float(np.std(bg_f))
    alpha = 0.85 if bg_std >= 22.0 else 0.0
    blended = cv2.addWeighted(flat, alpha, L, 1.0 - alpha, 0)
    if A is None:
        return blended
    return cv2.cvtColor(cv2.merge([blended, A, B]), cv2.COLOR_LAB2BGR)

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order_points(pts)
    wA, wB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    hA, hB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    W = max(int(wA), int(wB), 1)
    H = max(int(hA), int(hB), 1)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(_order_points(pts), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)

def perspective_rectify(img: np.ndarray, min_contour_area_ratio: float = 0.30,
                        approx_epsilon_ratio: float = 0.02) -> Optional[np.ndarray]:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    img_area = img.shape[0] * img.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_contour_area_ratio * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_epsilon_ratio * peri, True)
        if len(approx) == 4:
            return _four_point_transform(img, approx.reshape(4, 2))
    return None

def _estimate_skew_angle(img: np.ndarray) -> Tuple[float, str]:
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    H, W = gray_full.shape[:2]
    my, mx = int(H * 0.12), int(W * 0.12)
    gray = gray_full[my:H - my, mx:W - mx] if (H > 100 and W > 100) else gray_full

    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dil = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)), iterations=1)
    edges = cv2.Canny(dil, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=45, minLineLength=max(int(gray.shape[1] * 0.18), 35), maxLineGap=12)
    if lines is not None:
        h_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -25.0 <= ang <= 25.0:
                    h_angles.append(ang)
        if len(h_angles) >= 3 and abs(float(np.median(h_angles))) >= 1.8:
            return float(np.median(h_angles)), "hough"

    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in cnts:
        if cv2.contourArea(c) < 120:
            continue
        (_, _), (cw, ch), ang = cv2.minAreaRect(c)
        if cw < ch:
            cw, ch = ch, cw
            ang += 90.0
        if cw > 2.2 * max(ch, 1.0) and -25.0 <= ang <= 25.0:
            angles.append(ang)
    if len(angles) >= 3 and abs(float(np.median(angles))) >= 1.8:
        return float(np.median(angles)), "minAreaRect"
    return 0.0, "none"

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - w / 2
    M[1, 2] += nH / 2 - h / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def deskew(img: np.ndarray, mode: str = "rotate", **kw) -> Tuple[np.ndarray, str]:
    if mode == "perspective":
        warped = perspective_rectify(
            img,
            min_contour_area_ratio=kw.get("min_contour_area_ratio", 0.35),
            approx_epsilon_ratio=kw.get("approx_epsilon_ratio", 0.02),
        )
        if warped is not None:
            return warped, "perspective"
        return img, "none"
    angle, method_used = _estimate_skew_angle(img)
    if abs(angle) < 1.8:
        return img, "none"
    max_rot = float(kw.get("max_rotation_deg", 18.0))
    angle = float(np.clip(angle, -max_rot, max_rot))
    return rotate_image(img, angle), f"rotate_{method_used}({angle:.2f})"

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([clahe.apply(L), A, B]), cv2.COLOR_LAB2BGR)
    return clahe.apply(img)

def resize_for_ocr(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

@dataclass
class PreprocessResult:
    original: np.ndarray
    processed: np.ndarray
    steps: dict = field(default_factory=dict)

def preprocess_image(img: np.ndarray, cfg: dict) -> PreprocessResult:
    p = cfg["preprocessing"]
    out, steps = img.copy(), {"shadow_removal": False, "deskew": "off", "clahe": False}
    sr = p["shadow_removal"]
    if sr.get("enabled", True):
        out = remove_shadow(out, sr.get("bg_kernel_size", 21), sr.get("median_blur_ksize", 21), sr.get("mode", "divide"))
        steps["shadow_removal"] = True
    ds = p["deskew"]
    if ds.get("enabled", True):
        out, mode_used = deskew(
            out,
            mode=ds.get("mode", "auto"),
            min_contour_area_ratio=ds.get("min_contour_area_ratio", 0.30),
            approx_epsilon_ratio=ds.get("approx_epsilon_ratio", 0.02),
            max_rotation_deg=ds.get("max_rotation_deg", 15.0),
        )
        steps["deskew"] = mode_used
    cl = p["clahe"]
    if cl.get("enabled", True):
        out = apply_clahe(out, cl.get("clip_limit", 2.0), cl.get("tile_grid_size", 8))
        steps["clahe"] = True
    return PreprocessResult(original=img, processed=out, steps=steps)
