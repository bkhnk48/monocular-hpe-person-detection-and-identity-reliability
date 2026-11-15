"""
Chi phí của mạng time-expanded.

Ký hiệu (đặt tên theo configs/default.yaml, mục association):
  alpha  = weights.pose          trọng số chi phí pose (unary)
  gamma  = weights.temporal      trọng số chi phí chuyển động giữa 2 frame
  eta    = weights.displacement  trọng số C_V: độ dịch 2D giữa 2 detection liên tiếp
  beta   = weights.reprojection  trọng số C_2D: sai số 2D giữa detection và người GT trên ảnh
  mu     = weights.id_switch     C_ID: phạt khi một người đi từ body_id này sang body_id khác
                                 giữa 2 detection liên tiếp (dùng ID tracker của WHAM)
  C_FP, C_FN, C_disappear, C_reappear, chi phí biên

Chi phí unary của nút "detection d gán cho người k ở frame t":
    alpha * PA-MPJPE(d, k) / pose_scale  +  beta * E_2D(d, k) / s_2D  -  C_FP
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
    gap_power: float = 1.0      # temporal chia cho dt**gap_power (0 = bỏ qua dt)
    start_detected: float = 0.0
    start_missed: float = 0.0
    end_detected: float = 0.0
    end_missed: float = 0.0
    eta: float = 0.0            # weights.displacement (C_V; 0 = tắt)
    disp_scale: float = 0.05    # s_V: độ dịch 2D tính theo cỡ người trên ảnh
    beta: float = 0.0           # weights.reprojection (C_2D; 0 = tắt)
    reproj_scale: float = 0.05  # s_2D: sai số 2D so với GT, tính theo cỡ người GT trên ảnh
    mu: float = 0.0             # weights.id_switch (C_ID; 0 = tắt, không dùng body_id)

    @classmethod
    def from_config(cls, assoc: dict) -> "CostParams":
        w, c, s = assoc.get("weights", {}), assoc.get("costs", {}), assoc.get("scales", {})
        b = assoc.get("boundary", {})
        return cls(alpha=w.get("pose", 1.0), gamma=w.get("temporal", 0.5),
                   c_fp=c.get("false_positive", 0.8), c_fn=c.get("false_negative", 1.0),
                   c_disappear=c.get("disappear", 0.15), c_reappear=c.get("reappear", 0.20),
                   pose_scale_mm=s.get("pose_mm", 100.0), motion_scale_m=s.get("motion_m", 0.05),
                   gap_power=assoc.get("temporal", {}).get("gap_power", 1.0),
                   start_detected=b.get("start_detected", 0.0),
                   start_missed=b.get("start_missed", 0.0),
                   end_detected=b.get("end_detected", 0.0), end_missed=b.get("end_missed", 0.0),
                   eta=w.get("displacement", 0.0), disp_scale=s.get("displacement", 0.05),
                   beta=w.get("reprojection", 0.0), reproj_scale=s.get("reprojection", 0.05),
                   mu=w.get("id_switch", 0.0))


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


def motion_rms(res_a: np.ndarray, res_b: np.ndarray) -> float:
    """Độ thay đổi residual căn chỉnh giữa 2 frame (mét); 0 nếu thiếu khớp.

    Không giả định body_id giữ nguyên giữa 2 frame.
    """
    valid = np.isfinite(res_a).all(1) & np.isfinite(res_b).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((res_b[valid] - res_a[valid]) ** 2, axis=1))))


def motion_cost(rms_m, p: CostParams, gap=1):
    """gamma * RMS / (s_M * dt^gap_power).

    dt = số frame gốc giữa 2 lớp liên tiếp của cùng người (1 nếu liền nhau).
    Với gap_power = 1, chi phí đo tốc độ thay đổi residual trên mỗi frame, nên
    một khoảng trống 10 frame không bị phạt nặng như 10 lần một bước nhảy.
    """
    gap = np.maximum(np.asarray(gap, float), 1.0)
    return p.gamma * np.asarray(rms_m) / (p.motion_scale_m * gap ** p.gap_power)


def person_scale_2d(uv: np.ndarray) -> float:
    """Cỡ người trên ảnh (px): cạnh dài hơn của khung bao các khớp hợp lệ, tối thiểu 40 px."""
    ok = np.isfinite(uv).all(1)
    if not ok.any():
        return 40.0
    pts = uv[ok]
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 40.0))


def displacement_2d(uv_a, uv_b) -> float:
    """Độ dịch 2D giữa 2 detection: RMS độ dịch các khớp / cỡ người trung bình.

    Không dùng GT. Cùng một người ở 2 frame liền nhau dịch ít; nhảy sang người khác
    thì dịch cỡ khoảng cách giữa 2 người. 0 nếu thiếu toạ độ hoặc ít hơn 8 khớp
    hợp lệ (cùng quy ước với motion_rms).
    """
    if uv_a is None or uv_b is None:
        return 0.0
    valid = np.isfinite(uv_a).all(1) & np.isfinite(uv_b).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return 0.0
    rms = np.sqrt(np.mean(np.sum((uv_b[valid] - uv_a[valid]) ** 2, axis=1)))
    return float(rms / ((person_scale_2d(uv_a) + person_scale_2d(uv_b)) / 2.0))


def displacement_cost(disp, p: CostParams, gap=1):
    """C_V = eta * độ dịch / (s_V * dt^gap_power), chia dt giống motion_cost."""
    gap = np.maximum(np.asarray(gap, float), 1.0)
    return p.eta * np.asarray(disp, float) / (p.disp_scale * gap ** p.gap_power)


def id_switch_cost(switch, p: CostParams):
    """C_ID = mu * [body_id(a) != body_id(b)] cho cung Detected -> Detected.

    switch: True ở cung mà hai detection khác body_id (network.arc_id_switch).
    Một lần đổi body_id là một sự kiện (như C_disappear), nên không chia theo dt.
    Không dùng GT: body_id là ID tracker của WHAM, là đầu vào chứ không phải nhãn.
    """
    return p.mu * np.asarray(switch, float)


def reprojection_error_2d(uv_det, uv_gt) -> float:
    """E_2D: trung bình khoảng cách các khớp (px) giữa detection và người GT trên ảnh,
    chia cho cỡ người GT (person_scale_2d), nên không phụ thuộc độ phân giải video.

    Dùng GT (như PA-MPJPE). NaN nếu thiếu toạ độ hoặc ít hơn 8 khớp hợp lệ ở cả hai;
    reprojection_cost() coi NaN là không có bằng chứng (chi phí 0).
    """
    if uv_det is None or uv_gt is None:
        return float("nan")
    uv_det, uv_gt = np.asarray(uv_det, float), np.asarray(uv_gt, float)
    valid = np.isfinite(uv_det).all(1) & np.isfinite(uv_gt).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return float("nan")
    dist = np.linalg.norm(uv_det[valid] - uv_gt[valid], axis=1)
    return float(dist.mean() / person_scale_2d(uv_gt))


def reprojection_cost(err, p: CostParams):
    """C_2D = beta * E_2D / s_2D; E_2D = NaN (không đủ khớp, thiếu GT 2D) -> 0."""
    err = np.nan_to_num(np.asarray(err, float), nan=0.0)
    return p.beta * err / p.reproj_scale
