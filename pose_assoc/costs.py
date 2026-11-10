"""
Chi phí của mạng time-expanded.

Ký hiệu (đặt tên theo configs/default.yaml, mục association):
  alpha  = weights.pose       trọng số chi phí pose (unary)
  gamma  = weights.temporal   trọng số chi phí chuyển động giữa 2 frame
  C_FP, C_FN, C_disappear, C_reappear, chi phí biên

Chi phí unary của nút "detection d gán cho người k ở frame t":
    alpha * PA-MPJPE(d, k) / pose_scale  -  C_FP
(trừ C_FP vì một detection được dùng thì không còn là false positive; hằng số
 C_FP * tổng số detection được cộng lại khi báo cáo hàm mục tiêu.)

Chi phí unary của nút "người k bị bỏ lỡ ở frame t" (missed): C_FN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 12 khớp dùng chung giữa WHAM, MPI-INF-3DHP (13 khớp) và Harmony4D (COCO-17)
LIMBS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
MIN_VALID_JOINTS = 8


@dataclass
class CostParams:
    alpha: float = 1.0          # weights.pose
    gamma: float = 0.5          # weights.temporal
    c_fp: float = 0.8
    c_fn: float = 1.0
    c_disappear: float = 0.15
    c_reappear: float = 0.20
    pose_scale_mm: float = 100.0
    motion_scale_m: float = 0.05
    start_detected: float = 0.0
    start_missed: float = 0.0
    end_detected: float = 0.0
    end_missed: float = 0.0

    @classmethod
    def from_config(cls, assoc: dict) -> "CostParams":
        w, c, s = assoc.get("weights", {}), assoc.get("costs", {}), assoc.get("scales", {})
        b = assoc.get("boundary", {})
        return cls(alpha=w.get("pose", 1.0), gamma=w.get("temporal", 0.5),
                   c_fp=c.get("false_positive", 0.8), c_fn=c.get("false_negative", 1.0),
                   c_disappear=c.get("disappear", 0.15), c_reappear=c.get("reappear", 0.20),
                   pose_scale_mm=s.get("pose_mm", 100.0), motion_scale_m=s.get("motion_m", 0.05),
                   start_detected=b.get("start_detected", 0.0),
                   start_missed=b.get("start_missed", 0.0),
                   end_detected=b.get("end_detected", 0.0), end_missed=b.get("end_missed", 0.0))


def procrustes(pred: np.ndarray, gt: np.ndarray):
    """Căn pred theo gt (dịch + xoay + tỉ lệ). Trả về (PA-MPJPE mét, residual (J,3))."""
    valid = np.isfinite(pred).all(1) & np.isfinite(gt).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return np.nan, None
    X, Y = pred[valid], gt[valid]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    norm = np.sum(Xc ** 2)
    if norm < 1e-12:
        return np.nan, None
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
    R = U @ Vt
    aligned = S.sum() / norm * (Xc @ R) + Y.mean(0)
    residual = np.full_like(pred, np.nan)
    residual[valid] = aligned - Y
    return float(np.mean(np.linalg.norm(aligned - Y, axis=1))), residual


def detection_cost(pa_m: float, p: CostParams) -> float:
    return p.alpha * (pa_m * 1000.0) / p.pose_scale_mm - p.c_fp


def motion_cost(res_a: np.ndarray, res_b: np.ndarray, p: CostParams) -> float:
    """Độ thay đổi residual căn chỉnh giữa 2 frame (không giả định body_id giữ nguyên)."""
    valid = np.isfinite(res_a).all(1) & np.isfinite(res_b).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return 0.0
    rms = np.sqrt(np.mean(np.sum((res_b[valid] - res_a[valid]) ** 2, axis=1)))
    return p.gamma * rms / p.motion_scale_m


def transition_cost(a: dict, b: dict, p: CostParams) -> float:
    """a, b: nút ở 2 lớp liên tiếp của cùng một người."""
    if a["kind"] == "det" and b["kind"] == "det":
        return motion_cost(a["residual"], b["residual"], p)
    if a["kind"] == "det":
        return p.c_disappear
    if b["kind"] == "det":
        return p.c_reappear
    return 0.0
