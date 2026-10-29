import re
from typing import Dict, List, Tuple

def _edit_distance(seq1, seq2) -> int:
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def compute_cer(pred_text: str, gt_text: str) -> float:
    """Character Error Rate (CER) chuẩn hóa khoảng trắng."""
    gt = re.sub(r"\s+", " ", (gt_text or "").strip())
    pr = re.sub(r"\s+", " ", (pred_text or "").strip())
    if not gt:
        return 0.0 if not pr else 1.0
    return _edit_distance(list(pr), list(gt)) / max(len(gt), 1)

def compute_wer(pred_text: str, gt_text: str) -> float:
    """Word Error Rate (WER)."""
    gt_words = (gt_text or "").strip().split()
    pr_words = (pred_text or "").strip().split()
    if not gt_words:
        return 0.0 if not pr_words else 1.0
    return _edit_distance(pr_words, gt_words) / max(len(gt_words), 1)

def compute_entity_f1(preds: List[Dict[str, str]], gts: List[Dict[str, str]],
                      fields: List[str] = ("total_amount", "date", "invoice_id", "vendor")) -> Dict[str, dict]:
    """Tính Precision, Recall, F1-Score cho từng trường KIE và Micro-F1 tổng hợp."""
    report = {}
    tot_tp = tot_fp = tot_fn = 0
    for f in fields:
        tp = fp = fn = 0
        for p_dict, g_dict in zip(preds, gts):
            p_val = str(p_dict.get(f, "") or "").strip().lower()
            g_val = str(g_dict.get(f, "") or "").strip().lower()
            if not g_val and not p_val:
                continue
            if g_val and p_val == g_val:
                tp += 1
            elif p_val and not g_val:
                fp += 1
            elif g_val and not p_val:
                fn += 1
            else:
                fp += 1
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[f] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        tot_tp += tp; tot_fp += fp; tot_fn += fn
    m_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    m_rec  = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    m_f1   = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0
    report["micro_avg"] = {"precision": round(m_prec, 4), "recall": round(m_rec, 4), "f1": round(m_f1, 4)}
    return report
