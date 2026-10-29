import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List
from src.preprocessing.pipeline import resize_for_ocr, _four_point_transform

@dataclass
class TextBox:
    polygon: List[List[float]]
    text: str
    confidence: float

def _rget(r, key, default=None):
    if hasattr(r, "get"):
        try:
            v = r.get(key, None)
            if v is not None:
                return v
        except (KeyError, TypeError):
            pass
    if hasattr(r, "res") and isinstance(r.res, dict):
        v = r.res.get(key, None)
        if v is not None:
            return v
    try:
        return r[key]
    except (KeyError, TypeError, IndexError):
        return getattr(r, key, default)

def _is_empty(x) -> bool:
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        return False

class OCREngine:
    def __init__(self, lang="vi", use_textline_orientation=True,
                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                 text_det_limit_side_len=1280, text_det_limit_type="max",
                 drop_score=0.5, enable_mkldnn=False, **_):
        from paddleocr import PaddleOCR
        self.drop_score = float(drop_score)
        self.enable_mkldnn = enable_mkldnn
        common = dict(
            lang=lang,
            use_textline_orientation=use_textline_orientation,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
        )
        try:
            self.pipeline = PaddleOCR(enable_mkldnn=enable_mkldnn, **common)
        except (TypeError, ValueError):
            self.pipeline = PaddleOCR(**common)

    def _iter_results(self, raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return raw
        try:
            return list(raw)
        except TypeError:
            return [raw]

    def run(self, img: np.ndarray, max_side: int = 1280, verbose: bool = False) -> List[TextBox]:
        t0 = time.time()
        H0, W0 = img.shape[:2]
        small = resize_for_ocr(img, max_side)
        H1, W1 = small.shape[:2]
        sx, sy = W0 / W1, H0 / H1
        try:
            raw = self.pipeline.predict(small)
        except Exception as e:
            if verbose:
                print(f"  [OCR ERROR] {type(e).__name__}: {e}")
            return []

        out = []
        for r in self._iter_results(raw):
            texts  = _rget(r, "rec_texts")  or []
            scores = _rget(r, "rec_scores") or []
            polys  = _rget(r, "rec_polys") or _rget(r, "dt_polys") or []
            n = min(len(texts), len(scores), len(polys))
            for i in range(n):
                try:
                    s = float(scores[i])
                except Exception:
                    continue
                if s < self.drop_score or not str(texts[i]).strip():
                    continue
                poly = (np.asarray(polys[i], np.float32) * [sx, sy]).tolist()
                out.append(TextBox(polygon=poly, text=str(texts[i]).strip(), confidence=s))
        if verbose:
            print(f"  OCR run: {len(out)} boxes / {(time.time()-t0)*1000:.0f} ms (resized {W1}x{H1})")
        return out
