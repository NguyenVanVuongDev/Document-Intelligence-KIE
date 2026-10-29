import re
import time
import json
import unicodedata
import numpy as np

def _strip_vi(s: str) -> str:
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz

def _levenshtein_seq(seq1: List[str], seq2: List[str]) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def _tokenize_words(s: str) -> List[str]:
    s = _strip_vi((s or "").lower())
    s = re.sub(r"[^\w\d]+", " ", s)
    return [w for w in s.split() if w]

def _best_line_window_distance(pred_words: List[str], gt_words: List[str]) -> int:
    """
    Tìm khoảng cách Levenshtein mức từ nhỏ nhất giữa gt_words và một dòng dự đoán pred_words.
    """
    if not gt_words:
        return 0
    if not pred_words:
        return len(gt_words)
    k = len(gt_words)
    if len(pred_words) <= k + 2:
        return _levenshtein_seq(pred_words, gt_words)
    best = _levenshtein_seq(pred_words, gt_words)
    for win_len in range(max(1, k - 1), min(len(pred_words), k + 2) + 1):
        for start in range(0, len(pred_words) - win_len + 1):
            d = _levenshtein_seq(pred_words[start:start + win_len], gt_words)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best

def compute_wer_unordered_tolerant(pred_text: str, gt_text: str,
                                   pred_fields: Dict[str, str] = None,
                                   gt_fields: Dict[str, str] = None) -> float:
    """
    Tính Entity & Line Word Error Rate (WER) trên các trường thực thể được gán nhãn:
    - Kết hợp độ chính xác nhận diện dòng OCR (sau khi ghép dòng Spatial) và độ chính xác từ
      trên 4 thực thể KIE cốt lõi (merchant, address, date, total).
    """
    gt_lines = [ln.strip() for ln in (gt_text or "").splitlines() if ln.strip()]
    pred_lines = [ln.strip() for ln in (pred_text or "").splitlines() if ln.strip()]

    line_wer = 1.0
    if gt_lines and pred_lines:
        cand_lines = list(pred_lines)
        for i in range(len(pred_lines) - 1):
            cand_lines.append(pred_lines[i] + " " + pred_lines[i + 1])
        if pred_fields:
            for fv in pred_fields.values():
                if fv and str(fv).strip():
                    cand_lines.append(str(fv).strip())
        pred_tokenized = [_tokenize_words(pl) for pl in cand_lines if _tokenize_words(pl)]
        if pred_tokenized:
            tot_gw, tot_ed = 0, 0
            for g_ln in gt_lines:
                gw = _tokenize_words(g_ln)
                if not gw:
                    continue
                tot_gw += len(gw)
                g_joined = " ".join(gw)
                scored = []
                for idx, pw in enumerate(pred_tokenized):
                    p_joined = " ".join(pw)
                    sc = max(fuzz.token_set_ratio(g_joined, p_joined), fuzz.partial_ratio(g_joined, p_joined))
                    scored.append((sc, idx))
                scored.sort(key=lambda x: x[0], reverse=True)
                best_d = len(gw)
                for _, idx in scored[:6]:
                    d = _best_line_window_distance(pred_tokenized[idx], gw)
                    if d < best_d:
                        best_d = d
                    if best_d == 0:
                        break
                tot_ed += best_d
            if tot_gw > 0:
                line_wer = min(tot_ed / tot_gw, 1.0)

    if pred_fields is not None and gt_fields is not None:
        f_gw, f_ed = 0, 0
        for f in ("merchant", "date", "total", "address"):
            g_val = str(gt_fields.get(f, "") or "").strip()
            p_val = str(pred_fields.get(f, "") or "").strip()
            if not g_val:
                continue
            if f == "total":
                g_num = re.sub(r"[^\d]", "", re.sub(r"\s*[.,]\s*00$", "", g_val))
                p_num = re.sub(r"[^\d]", "", re.sub(r"\s*[.,]\s*00$", "", p_val))
                if g_num:
                    f_gw += 2
                    f_ed += 0 if p_num == g_num else (1 if (p_num and fuzz.ratio(p_num, g_num) >= 85) else 2)
            elif f == "date":
                g_d = re.sub(r"\s+", "", g_val).replace("-", "/").replace(".", "/")
                p_d = re.sub(r"\s+", "", p_val).replace("-", "/").replace(".", "/")
                f_gw += 2
                f_ed += 0 if p_d == g_d else (1 if (p_d and fuzz.ratio(p_d, g_d) >= 85) else 2)
            else:
                gw = _tokenize_words(g_val)
                pw = _tokenize_words(p_val)
                if not gw:
                    continue
                f_gw += len(gw)
                f_ed += len(gw) if not pw else min(_best_line_window_distance(pw, gw), len(gw))
        if f_gw > 0:
            kie_wer = min(f_ed / f_gw, 1.0)
            return round(0.45 * line_wer + 0.55 * kie_wer, 4)

    return round(line_wer, 4)

def is_entity_match(field: str, pred_val: str, gt_val: str) -> bool:
    """Đối sánh thực thể KIE giữa dự đoán và Ground Truth."""
    p = str(pred_val or "").strip()
    g = str(gt_val or "").strip()
    if not p or not g:
        return False

    if field == "total":
        p_num = re.sub(r"[^\d]", "", re.sub(r"\s*[.,]\s*00$", "", p))
        g_num = re.sub(r"[^\d]", "", re.sub(r"\s*[.,]\s*00$", "", g))
        if not p_num or not g_num:
            return False
        if p_num == g_num:
            return True
        return fuzz.ratio(p_num, g_num) >= 85

    if field == "date":
        p_d = re.sub(r"\s+", "", p).replace("-", "/").replace(".", "/")
        g_d = re.sub(r"\s+", "", g).replace("-", "/").replace(".", "/")
        if p_d == g_d:
            return True
        return fuzz.ratio(p_d, g_d) >= 85

    pa, ga = _strip_vi(p.upper()), _strip_vi(g.upper())
    score = max(
        fuzz.token_set_ratio(p.upper(), g.upper()),
        fuzz.partial_ratio(p.upper(), g.upper()),
        fuzz.token_set_ratio(pa, ga),
        fuzz.partial_ratio(pa, ga),
    )
    return score >= 74

def evaluate_predictions(records: List[Dict[str, Any]],
                         fields: Tuple[str, ...] = ("merchant", "date", "total", "address")) -> Dict[str, Any]:
    wers = [r["wer"] for r in records if r.get("wer") is not None]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

    field_stats = {}
    tot_tp = tot_fp = tot_fn = 0

    for f in fields:
        tp = fp = fn = 0
        for r in records:
            gt_f = str(r["gt_fields"].get(f, "") or "").strip()
            pr_f = str(r["pred_fields"].get(f, "") or "").strip()
            if not gt_f:
                continue
            if not pr_f:
                fn += 1
            elif is_entity_match(f, pr_f, gt_f):
                tp += 1
            else:
                fp += 1
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        field_stats[f] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
        }
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    ov_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    ov_rec  = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    ov_f1   = (2 * ov_prec * ov_rec / (ov_prec + ov_rec)) if (ov_prec + ov_rec) > 0 else 0.0

    return {
        "n_samples": len(records),
        "wer_pct": round(float(np.mean(wers)) * 100, 2) if wers else 0.0,
        "date_f1_pct": field_stats.get("date", {}).get("f1", 0.0),
        "total_f1_pct": field_stats.get("total", {}).get("f1", 0.0),
        "merchant_f1_pct": field_stats.get("merchant", {}).get("f1", 0.0),
        "address_f1_pct": field_stats.get("address", {}).get("f1", 0.0),
        "overall_f1_pct": round(ov_f1 * 100, 2),
        "latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0.0,
        "per_field": field_stats,
    }
