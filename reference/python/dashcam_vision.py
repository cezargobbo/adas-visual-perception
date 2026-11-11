#!/usr/bin/env python3
"""
Dashcam vision pipeline:
  1. Semantic segmentation (Cityscapes, 19 classes) with SegFormer (NVIDIA, HuggingFace): B5 or B1.
  2. Object detection + tracking (people, vehicles, traffic lights) with a COCO-pretrained YOLO (Ultralytics).
  3. Lane detection (on by default, --no-lane disables; Polar R-CNN) + ego-lane tracking with temporal extrapolation through
     intersections; the ego lane then replaces the fixed trapezoid as lead-vehicle corridor.
  5. ADAS warnings (on by default): FCW (TTC from bbox scale expansion of the lead vehicle) and LDW (lane departure,
     very conservative arming logic); events go to <prefix>_alerts.csv.
  4. Lead-vehicle selection (the vehicle directly ahead in the ego lane) + monocular distance estimate,
     drawn as a line from the bottom of its bbox to the camera (bottom-centre of the image).

Usage:
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 -o outputs/sp1          # -> outputs/sp1_*.mp4 (-o explicit wins)
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 -o outputs/sp1_fast --profile fast            # light models (real time)
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 -o outputs/sp1_det --profile fast --no-seg    # detection only
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 -o outputs/sp1_seg --profile fast --no-det    # segmentation only
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 --profile fast --no-seg --show   # live window only (no files)
  .venv/bin/python dashcam_vision.py sp1_crop.mp4 -o outputs/test --start 120 --duration 10   # quick test

Distance estimation (no camera calibration available, so it is an approximation):
  * width cue : Z = f * W_real / w_px        (W_real = typical width of the class, seen from behind)
  * ground cue: Z = f * cam_height / (y_bottom - y_horizon)   (flat-road pinhole model)
  Both are fused and smoothed over time per track. f comes from --hfov. Tune --hfov, --cam-height and
  --horizon if you know the camera setup.
"""
import argparse
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- Cityscapes
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
CITYSCAPES_PALETTE_RGB = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)
PALETTE_BGR = CITYSCAPES_PALETTE_RGB[:, ::-1].copy()

# ----------------------------------------------------------------------------- COCO classes we keep
COCO_KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light"}
VEHICLE_IDS = {2, 3, 5, 7}              # candidates for "lead vehicle"
REAL_WIDTH_M = {2: 1.80, 3: 0.80, 5: 2.55, 7: 2.40}   # typical rear width per class
BOX_COLORS = {0: (60, 20, 220), 1: (230, 0, 230), 2: (255, 160, 0), 3: (255, 255, 0),
              5: (0, 200, 255), 7: (0, 140, 255), 9: (0, 255, 0)}


# ============================================================================= Segmentation
class Segmenter:
    def __init__(self, model_id, device, infer_w, infer_h, fp16=True):
        from transformers import SegformerForSemanticSegmentation
        self.device = device
        self.dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device, self.dtype).eval()
        self.size = (infer_h, infer_w)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.palette = torch.from_numpy(PALETTE_BGR).to(device)

    @torch.inference_mode()
    def __call__(self, frame_bgr):
        """Returns the class-id map (H, W) as a GPU uint8 tensor, at full frame resolution."""
        h, w = frame_bgr.shape[:2]
        x = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        x = x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        logits = self.model(pixel_values=x).logits                      # (1, 19, h/4, w/4)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].to(torch.uint8)

    @torch.inference_mode()
    def overlay(self, frame_bgr, seg_map, alpha):
        """Blend the Cityscapes colours over the frame on the GPU. Returns (BGR uint8 image, set of class ids)."""
        f = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        color = self.palette[seg_map.long()]
        out = (f.half() * (1 - alpha) + color.half() * alpha).round().to(torch.uint8).cpu().numpy()
        ids = set(torch.unique(seg_map[::8, ::8]).tolist())
        return out, ids


def draw_legend(img, present_ids, x0=10, y0=10):
    ids = [i for i in range(len(CITYSCAPES_CLASSES)) if i in present_ids]
    if not ids:
        return
    row_h, box_w = 22, 150
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x0 + box_w, y0 + row_h * len(ids) + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)
    for k, cid in enumerate(ids):
        y = y0 + k * row_h
        cv2.rectangle(img, (x0, y + 3), (x0 + 16, y + 17), tuple(int(c) for c in PALETTE_BGR[cid]), -1)
        cv2.putText(img, CITYSCAPES_CLASSES[cid], (x0 + 22, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================= Traffic light state
def _topk_mean(a, frac=0.25):
    a = np.sort(a.ravel())
    k = max(1, int(round(frac * a.size)))
    return float(a[-k:].mean())


def traffic_light_state(frame, box, thr=14.0):
    """Colour of the lit lamp: 'red' / 'yellow' / 'green' / None.
    Uses WHERE the lit lamp is (vertical housing: red top, yellow middle, green bottom; horizontal: red/green at
    the ends) and its colour measured RELATIVE to the other lamp cells of the same housing, so it copes with
    washed-out / pink LEDs, low-resolution video and painted (e.g. yellow) housings."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return None
    h, w = crop.shape[:2]
    vertical = h >= 1.3 * w
    if not vertical and w < 1.3 * h:
        return None                                       # square box: side view / partial housing
    c = crop.astype(np.float32)
    if vertical:
        m = int(round(0.15 * w))
        c = c[:, m:w - m] if w - 2 * m >= 2 else c        # drop the background at the sides
    else:
        m = int(round(0.15 * h))
        c = (c[m:h - m, :] if h - 2 * m >= 2 else c).transpose(1, 0, 2)   # long axis first
    B, G, R = c[..., 0], c[..., 1], c[..., 2]
    V = c.max(axis=2)
    feats = {"red": R - G, "yellow": np.minimum(R, G) - B, "green": G - R}
    L = c.shape[0]
    cells = [slice(int(k * L / 3), max(int((k + 1) * L / 3), int(k * L / 3) + 1)) for k in range(3)]
    bright = [_topk_mean(V[sl]) for sl in cells]
    f = {col: [_topk_mean(F[sl]) for sl in cells] for col, F in feats.items()}
    expected = {"red": (0,) if vertical else (0, 2), "yellow": (1,), "green": (2,) if vertical else (2, 0)}
    scores = {}
    for col, ks in expected.items():
        best = -1e9
        for k in ks:
            others = [j for j in range(3) if j != k]
            colour = f[col][k] - max(f[col][j] for j in others)          # colour excess vs. the other cells
            light = bright[k] - np.mean([bright[j] for j in others])     # the lit cell is also brighter
            best = max(best, colour + 0.25 * light)
        scores[col] = best
    st = max(scores, key=scores.get)
    return st if scores[st] >= thr else None


# ============================================================================= Ego lane (Polar R-CNN + temporal tracking)
LANES_THR_DEFAULT = 0.44          # = CONF_NMSFREE in lanes_polar.py (duplicated here so the model is not imported with --no-lane)
LANE_COLOR = (70, 220, 70)        # ego lane (BGR)
LANE_OTHER_COLOR = (255, 210, 130)  # other detected lanes (BGR)


def lane_x_at(lane, ys, H):
    """x(y) of a detected lane sampled at rows `ys`: polynomial (degree <= 2) inside the observed y-range,
    straight tangent continuation outside it (so the lane can be extrapolated down to the image bottom).
    Returns None when the lane is too short / does not reach the lower part of the image."""
    x, y = lane[:, 0].astype(np.float64), lane[:, 1].astype(np.float64)
    if len(y) < 5 or np.ptp(y) < 0.15 * H or y.max() < 0.75 * H or y.min() > 0.85 * H:
        return None
    c = np.polyfit(y, x, 2 if np.ptp(y) > 0.25 * H else 1)
    dc = np.polyder(c)
    yc = np.clip(ys, y.min(), y.max())
    slope = np.clip(np.polyval(dc, yc), -2.0, 2.0)
    return np.polyval(c, yc) + slope * (ys - yc)


class EgoLaneTracker:
    """Picks the ego-lane boundaries from the detected lanes and tracks them over time.

    States: 'detected' (both boundaries seen), 'partial' (one seen, the other inferred from the last known
    lane-width profile), 'estimated' (nothing seen: the last good lane is HELD for `hold_s` seconds, fading
    out; when driving straight the ego lane stays roughly fixed in the image) and 'none'.

    Lane width at the bottom row, pinhole + flat road: w_px = w_m * (H - y_horizon) / cam_height
    (the focal length cancels out, so hfov does not matter). For 2.7-4.0 m, cam 1.5 m, horizon 0.59 H
    at 1080p that is 797-1181 px; with a tolerance (-25% / +10%) for horizon/camera uncertainty the accepted
    range is ~600-1300 px. On sp1_crop_30fps.mp4 the median width of the nearest left/right lane pair is
    902 px (p25 773, p5 620), consistent with this."""

    def __init__(self, W, H, fps, cam_height, horizon_frac, hold_s, n_rows=40):
        self.W, self.H = W, H
        self.y_hor = horizon_frac * H
        self.ys = np.linspace(max(horizon_frac + 0.04, 0.63) * H, H - 1, n_rows)
        self.t_row = (self.ys - self.y_hor) / (H - 1 - self.y_hor)           # relative lane width per row (perspective)
        scale = (H - 1 - self.y_hor) / cam_height                              # px per metre at the bottom row
        self.wmin, self.wmax = 2.7 * scale * 0.75, 4.0 * scale * 1.10
        self.wdef = 3.4 * scale * self.t_row                                   # default width profile
        self.gate = 0.07 * W                                                   # jump gate at the bottom rows (px)
        self.persist = max(3, int(0.3 * fps))                                  # frames a "jump" must persist
        self.hold_frames = max(1, int(hold_s * fps))
        self.alpha = 0.35
        self.reset()

    def reset(self):
        self.xl = self.xr = None
        self.wprof = self.wdef.copy()
        self.state, self.since, self.fade = "none", 0, 0.0
        self.obs_l = self.obs_r = False
        self.pend = {"l": [None, 0], "r": [None, 0]}
        self.free = {"l": 0, "r": 0}
        self.glitch = False                        # this frame: jump-gate rejection / re-acquisition / implausible lane
        self.raw_l = self.raw_r = None             # accepted raw observations of this frame (before the EMA)

    # ---------------------------------------------------------------- helpers
    def _gate(self, side, obs):
        """Jump gate. Returns True if the observation is accepted for this side."""
        trk = self.xl if side == "l" else self.xr
        if trk is None or self.free[side] > 0:
            return True
        n = len(self.ys) // 2
        if np.mean(np.abs(obs[n:] - trk[n:])) <= self.gate:
            self.pend[side] = [None, 0]
            return True
        self.glitch = True
        p, k = self.pend[side]
        k = k + 1 if (p is not None and np.mean(np.abs(obs[n:] - p[n:])) <= self.gate) else 1
        self.pend[side] = [obs, k]
        if k >= self.persist:                       # persistent new position -> re-acquire (blended by the EMA)
            self.pend[side] = [None, 0]
            self.free[side] = self.persist
            return True
        return False

    def _pick(self, cands):
        """cands: list of x(ys) arrays, one per lane -> (left, right) observations (arrays or None)."""
        W = self.W
        left = [c for c in cands if c[-1] < W / 2]
        right = [c for c in cands if c[-1] >= W / 2]
        best, best_cost = None, 1e18
        for l in left:
            for r in right:
                w = r - l
                if not (self.wmin <= w[-1] <= self.wmax):
                    continue
                rel = w / w[-1]                     # must follow the perspective profile (no crossing / diverging)
                if np.any(rel < 0.4 * self.t_row - 0.02) or np.any(rel > 1.8 * self.t_row + 0.04):
                    continue
                cost = (abs(l[-1] - self.xl[-1]) + abs(r[-1] - self.xr[-1])) if self.xl is not None else w[-1]
                if cost < best_cost:
                    best, best_cost = (l, r), cost
        if best is not None:
            return best

        def single(lst, trk):                      # no valid pair: one boundary only
            if not lst:
                return None
            if trk is not None:
                c = min(lst, key=lambda a: abs(a[-1] - trk[-1]))
                return c if abs(c[-1] - trk[-1]) < 2.0 * self.gate else None
            c = min(lst, key=lambda a: abs(a[-1] - W / 2))
            return c if abs(c[-1] - W / 2) < 0.85 * self.wmax else None
        return single(left, self.xl), single(right, self.xr)

    def _plausible(self, xl, xr):
        """Bottom-row sanity: lane width in range and lane centre not far from the image centre."""
        w, c = xr[-1] - xl[-1], 0.5 * (xr[-1] + xl[-1])
        return self.wmin <= w <= self.wmax and abs(c - self.W / 2) <= 0.18 * self.W

    # ---------------------------------------------------------------- per frame
    def update(self, lanes):
        self.glitch = False
        cands = [xs for xs in (lane_x_at(p, self.ys, self.H) for p in lanes) if xs is not None]
        ol, orr = self._pick(cands)
        for s in self.free:
            self.free[s] = max(0, self.free[s] - 1)
        if ol is not None and not self._gate("l", ol):
            ol = None
        if orr is not None and not self._gate("r", orr):
            orr = None
        a = self.alpha
        nxl = nxr = nw = None
        if ol is not None and orr is not None:
            first = self.xl is None
            nxl = ol if first else (1 - a) * self.xl + a * ol
            nxr = orr if first else (1 - a) * self.xr + a * orr
            nw = (orr - ol) if first else 0.9 * self.wprof + 0.1 * (orr - ol)
        elif ol is not None or orr is not None:
            if ol is not None:
                nxl = ol if self.xl is None else (1 - a) * self.xl + a * ol
                nxr = nxl + self.wprof
            else:
                nxr = orr if self.xr is None else (1 - a) * self.xr + a * orr
                nxl = nxr - self.wprof
        if nxl is not None and not self._plausible(nxl, nxr):   # drifting / leaving the image -> treat as a miss
            nxl = None
            ol = orr = None
            self.glitch = True
        self.raw_l, self.raw_r = ol, orr
        self.obs_l, self.obs_r = ol is not None, orr is not None
        if nxl is not None:
            self.xl, self.xr = nxl, nxr
            if nw is not None:
                self.wprof = nw
            both = ol is not None and orr is not None
            self.state, self.since, self.fade = ("detected" if both else "partial"), 0, 1.0
        else:
            self.since += 1
            if self.xl is None or self.since > self.hold_frames:
                self.reset()
            else:
                self.state = "estimated"
                self.fade = max(0.0, 1.0 - self.since / self.hold_frames)
        return self.state

    @property
    def available(self):
        return self.state != "none"

    def contains(self, box, margin=0.10):
        """Is the bottom-centre of `box` between the ego-lane boundaries at its y_bottom (+ margin)?"""
        x1, y1, x2, y2 = box
        if y2 <= self.y_hor + 5:
            return False
        cx = 0.5 * (x1 + x2)
        y = float(np.clip(y2, self.ys[0], self.ys[-1]))
        xl, xr = np.interp(y, self.ys, self.xl), np.interp(y, self.ys, self.xr)
        m = max(margin * (xr - xl), 0.01 * self.W)
        return xl - m <= cx <= xr + m


def draw_dashed(img, pts, color, thick):
    for i in range(len(pts) - 1):
        if i % 4 < 2:
            cv2.line(img, tuple(int(v) for v in pts[i]), tuple(int(v) for v in pts[i + 1]), color, thick, cv2.LINE_AA)


def draw_lanes(img, lanes, ego, ldw_side=None, ldw_armed=None):
    """All detected lanes as thin polylines + ego lane as translucent polygon with boundary lines
    (solid = observed, dashed = inferred/held; alpha fades while the lane is only held)."""
    H = img.shape[0]
    for p in lanes:
        cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], False, LANE_OTHER_COLOR, 2, cv2.LINE_AA)
    if ego is None or not ego.available:
        return
    fade = ego.fade if ego.state == "estimated" else 1.0
    yt = int(ego.ys[0])
    pl = np.stack([ego.xl, ego.ys], 1)
    pr = np.stack([ego.xr, ego.ys], 1)
    poly = np.concatenate([pl, pr[::-1]]).round().astype(np.int32)
    roi = img[yt:]
    ov = roi.copy()
    cv2.fillPoly(ov, [poly - np.array([0, yt], np.int32)], LANE_COLOR)
    a = 0.28 * fade
    cv2.addWeighted(ov, a, roi, 1 - a, 0, dst=roi)
    line_col = tuple(int(c * (0.35 + 0.65 * fade)) for c in LANE_COLOR)
    for pts, seen, side in ((pl, ego.obs_l, "left"), (pr, ego.obs_r, "right")):
        if ldw_side == side:                                   # boundary being approached (LDW)
            cv2.polylines(img, [pts.round().astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.polylines(img, [pts.round().astype(np.int32).reshape(-1, 1, 2)], False, (0, 100, 255), 8, cv2.LINE_AA)
        elif ego.state == "detected" or (ego.state == "partial" and seen):
            cv2.polylines(img, [pts.round().astype(np.int32).reshape(-1, 1, 2)], False, line_col, 4, cv2.LINE_AA)
        else:
            draw_dashed(img, pts.round(), line_col, 4)
    put_label(img, f"lane: {ego.state}", 12, H - 14, line_col, 0.6, 2)
    if ldw_armed is not None:                                  # discreet LDW status
        put_label(img, "LDW: armed" if ldw_armed else "LDW: disarmed", 12, H - 46,
                  (90, 200, 90) if ldw_armed else (150, 150, 150), 0.5, 1)


# ============================================================================= Lead vehicle + distance
class LeadVehicleEstimator:
    def __init__(self, W, H, hfov_deg, cam_height, horizon_frac, corridor_bottom, corridor_top, hold_frames):
        self.W, self.H = W, H
        self.f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cam_h = cam_height
        self.y_hor = horizon_frac * H
        self.cb, self.ct = corridor_bottom * W, corridor_top * W
        self.hold_frames = hold_frames
        self.lead_id, self.missing = None, 0
        self.hist = defaultdict(lambda: deque(maxlen=9))
        self.ema = {}

    def corridor_halfwidth(self, y):
        t = np.clip((y - self.y_hor) / (self.H - self.y_hor), 0.0, 1.0)
        return self.ct + t * (self.cb - self.ct)

    def in_corridor(self, box, lane=None):
        if lane is not None and lane.available:       # ego lane (detected / partial / held) instead of the trapezoid
            return lane.contains(box)
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        if y2 <= self.y_hor + 5:
            return False
        return abs(cx - self.W / 2) <= self.corridor_halfwidth(y2)

    def distance(self, box, cls):
        x1, y1, x2, y2 = box
        w_px = max(x2 - x1, 1.0)
        z_width = self.f * REAL_WIDTH_M.get(cls, 1.8) / w_px
        dy = y2 - self.y_hor
        z_ground = self.f * self.cam_h / dy if dy > 8 else None
        # bbox cut by the image border -> width cue unreliable
        cut = x1 <= 2 or x2 >= self.W - 2
        if z_ground is None:
            return z_width
        if cut:
            return z_ground
        return 0.6 * z_width + 0.4 * z_ground

    def update(self, dets, lane=None):
        """dets: list of (track_id, cls, box); lane: optional EgoLaneTracker (used as corridor when available).
        Returns (track_id, cls, box, dist_m) or None."""
        cands = [d for d in dets if d[1] in VEHICLE_IDS and self.in_corridor(d[2], lane)]
        best = max(cands, key=lambda d: d[2][3]) if cands else None
        # hysteresis: keep the current lead unless another candidate is clearly closer
        if self.lead_id is not None:
            cur = next((d for d in cands if d[0] == self.lead_id), None)
            if cur is not None and best is not None and best[2][3] - cur[2][3] < 0.03 * self.H:
                best = cur
        if best is None:
            self.missing += 1
            if self.missing > self.hold_frames:
                self.lead_id = None
            return None
        tid, cls, box = best
        if tid != self.lead_id:
            self.ema.pop(self.lead_id, None)
        self.lead_id, self.missing = tid, 0
        z = self.distance(box, cls)
        h = self.hist[tid]
        h.append(z)
        zm = float(np.median(h))
        self.ema[tid] = zm if tid not in self.ema else 0.8 * self.ema[tid] + 0.2 * zm
        return tid, cls, box, self.ema[tid]


# ============================================================================= ADAS: FCW + LDW
def theil_sen(t, y):
    """Robust slope dy/dt (median of pairwise slopes)."""
    t, y = np.asarray(t, np.float64), np.asarray(y, np.float64)
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 1e-6
    return float(np.median((y[j] - y[i])[ok] / dt[ok])) if ok.any() else 0.0


class FCWMonitor:
    """Forward Collision Warning from the scale expansion of the lead-vehicle bbox (calibration-free TTC).

    r = d ln(w) / dt  (Theil-Sen over `win` s)  ->  TTC = 1 / r   (r > 0 = approaching). Cross-check: the smoothed
    distance must also be shrinking (closing speed >= 0.5 m/s, distance-TTC < 2.5 x threshold), the bbox height must
    expand consistently with the width and the implied closing speed (dist * r) must be <= 14 m/s. Gates: same lead
    id for >= 0.5 s, bbox width >= 40 px, distance < 40 m, bbox not cut by the image border, bbox centre within
    1.0 box width of the image centre (a parked car beside the lane that we just pass is not a collision course).
    Debounce: condition for 0.3 s to trigger, 0.5 s without it to release. Levels: 'warning' / 'critical'."""

    def __init__(self, W, H, fps, ttc_alert, ttc_crit, win=0.8):
        self.W, self.H, self.fps = W, H, fps
        self.ta, self.tc, self.win = ttc_alert, ttc_crit, win
        self.on_n, self.off_n = max(1, int(0.3 * fps)), max(1, int(0.5 * fps))
        self.reset()
        self.active, self.level, self.cnt_on, self.cnt_off = False, None, 0, 0

    def reset(self):
        self.hist = deque()                     # (t, ln w, z, ln h)
        self.lead_id, self.since, self.miss = None, None, 0
        self.ttcs = deque(maxlen=5)

    def update(self, t, lead):
        """-> dict(ttc, active, level, event) ; event = None | 'warning' | 'critical' (newly raised level)."""
        ttc = None
        if lead is None:
            self.miss += 1
            if self.miss > 0.3 * self.fps:
                self.reset()
        else:
            tid, cls, box, dist = lead
            self.miss = 0
            if tid != self.lead_id:
                self.reset()
                self.lead_id, self.since = tid, t
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cut = x1 <= 2 or x2 >= self.W - 2 or y2 >= self.H - 2 or y1 <= 2
            # off-path (parked car beside the lane that we pass): bbox centre too far from the image centre in box widths
            off_path = abs(0.5 * (x1 + x2) - self.W / 2) > 1.0 * w
            if cut or w < 40 or h < 10 or off_path:
                self.hist.clear()               # unreliable scale -> restart the window
            else:
                self.hist.append((t, math.log(w), dist, math.log(h)))
                while self.hist and t - self.hist[0][0] > self.win:
                    self.hist.popleft()
                if (len(self.hist) >= max(8, int(0.55 * self.win * self.fps)) and t - self.since >= 0.5
                        and dist < 40.0 and self.hist[-1][0] - self.hist[0][0] >= 0.6 * self.win):
                    ts, lw, zs, lh = zip(*self.hist)
                    r = theil_sen(ts, lw)
                    rh = theil_sen(ts, lh)              # the height must expand consistently (turning vehicles only widen)
                    vz = -theil_sen(ts, zs)             # closing speed from the distance track (m/s)
                    if (r > 0.03 and vz >= 0.5 and dist / vz < 2.5 * self.ta and 0.5 * r <= rh <= 2.0 * r + 0.05
                            and dist * r <= 14.0):      # implied closing speed <= 50 km/h (bbox noise otherwise)
                        ttc = 1.0 / r
        if ttc is not None:
            self.ttcs.append(ttc)
            ttc = float(np.median(self.ttcs))
        else:
            self.ttcs.clear()
        cond = ttc is not None and ttc < self.ta
        lvl = None if not cond else ("critical" if ttc < self.tc else "warning")
        event = None
        if cond:
            self.cnt_on, self.cnt_off = self.cnt_on + 1, 0
            if self.cnt_on >= self.on_n:
                if not self.active or (lvl == "critical" and self.level != "critical"):
                    event = lvl
                self.active = True
                if lvl == "critical" or self.level is None:
                    self.level = lvl
        else:
            self.cnt_on, self.cnt_off = 0, self.cnt_off + 1
            if self.active and self.cnt_off >= self.off_n:
                self.active, self.level = False, None
        return dict(ttc=ttc, active=self.active, level=self.level if self.active else None, event=event)


class LDWMonitor:
    """Lane Departure Warning with a 'robustness first' arming logic (a missed alert is better than a false one).

    Lateral position: camera centre x = W/2 (camera assumed centred on the car), measured at row `row` (~0.93 H);
    lane width there = lane_w_m (3.5 m assumed; wider than most urban lanes, so the metric offsets err on the safe side) -> distance camera->left boundary DL (right: DR = lane_w - DL);
    wheel-to-line distance d = D - car_width / 2. Approach speed = -dD/dt (Theil-Sen over 0.5 s), TLC = d / speed.

    Armed only when EVERY frame of the last `arm_s` s was 'clean': state detectada with both boundaries observed,
    no jump-gate rejection / re-acquisition, raw observation agrees with the tracked boundary, width (normalised to the bottom row) inside the
    plausible range and stable (CV < 5 %), straight and mutually consistent boundaries (lane lines converge at the
    horizon, vanishing point near the image centre), lateral speed < 1.5 m/s. Any failure disarms at once.
    Alert: armed and d < margin + 0.3 * speed (allowance grows with the approach speed) and speed >= 0.15 m/s and TLC < 1 s and net drift >= 0.25 m over the last 1 s, held 0.3 s (debounce), release after 0.5 s,
    cooldown 2.5 s per side. A glitch (the tracker swaps boundaries on a real line crossing) while/just after an
    alert counts as a crossing and suppresses LDW for 2 s more."""

    def __init__(self, W, H, fps, trk, car_width=1.8, margin=0.25, lane_w_m=3.5, horizon_frac=0.59,
                 arm_s=1.0, fit_s=0.5):
        self.W, self.H, self.fps, self.trk = W, H, fps, trk
        self.row = int(np.argmin(np.abs(trk.ys - 0.93 * H)))
        self.car_w, self.margin, self.lane_w = car_width, margin, lane_w_m
        self.hor = horizon_frac * H
        self.arm_s, self.fit_s = arm_s, fit_s
        self.on_n, self.off_n = max(1, int(0.3 * fps)), max(1, int(0.5 * fps))
        self.cool_s, self.cross_s, self.vmin, self.vmax = 2.5, 2.0, 0.15, 1.5
        self.hist = deque()                     # (t, DL, w) of consecutive clean frames
        self.cnt = {"left": 0, "right": 0}
        self.off = {"left": 0, "right": 0}
        self.cool = {"left": -1e9, "right": -1e9}
        self.active = None
        self.last_active_t = -1e9
        self.suppress_until = -1e9
        self.armed, self.fail = False, "start"
        self.m = {}
        self.stats = defaultdict(int)           # frames per disarm reason (tuning / info)
        self.n_cross = 0

    # ---------------------------------------------------------------- per-frame cleanliness
    def _check(self):
        trk, r = self.trk, self.row
        if trk.state != "detected" or not (trk.obs_l and trk.obs_r):
            return "state"
        if trk.glitch or trk.free["l"] > 0 or trk.free["r"] > 0:
            return "glitch"
        n = len(trk.ys) // 2
        if (np.mean(np.abs(trk.raw_l[n:] - trk.xl[n:])) > 40 or np.mean(np.abs(trk.raw_r[n:] - trk.xr[n:])) > 40):
            return "obs_vs_trk"
        w = trk.xr - trk.xl
        if not (trk.wmin <= w[r] / trk.t_row[r] <= trk.wmax):      # wmin/wmax are bottom-row widths; normalise row r
            return "width"
        ys = trk.ys
        p = np.polyfit(ys, w, 1)
        if p[0] <= 0:
            return "convergence"
        y0 = -p[1] / p[0]                                           # row where the lane width vanishes
        if not (self.hor - 0.07 * self.H <= y0 <= self.hor + 0.07 * self.H):
            return "convergence"
        if np.max(np.abs(np.polyval(p, ys) - w)) > 0.04 * w[r]:
            return "width_profile"
        xv = []
        self.m = {"y0": y0, "wres": float(np.max(np.abs(np.polyval(p, ys) - w)) / w[r])}
        for k, x in zip("lr", (trk.xl, trk.xr)):
            q = np.polyfit(ys, x, 1)
            self.m["res" + k] = float(np.max(np.abs(np.polyval(q, ys) - x)))
            self.m["sl" + k] = float(q[0])
            if np.max(np.abs(np.polyval(q, ys) - x)) > 0.0065 * self.W:     # ~12 px
                return "curve"
            xv.append(np.polyval(q, y0))
        if abs(0.5 * (xv[0] + xv[1]) - self.W / 2) > 0.2 * self.W or abs(xv[0] - xv[1]) > 0.08 * self.W:
            return "vanishing_point"
        return None

    def update(self, t, lane_ok=True):
        """-> dict(armed, side (active), event (newly triggered side or None), d, v, tlc)."""
        trk, r = self.trk, self.row
        fail = self._check() if lane_ok else "state"
        DL = vL = None
        self.last_v = None
        if fail is None:
            w = float(trk.xr[r] - trk.xl[r])
            DL = (self.W / 2 - float(trk.xl[r])) / w * self.lane_w
            self.hist.append((t, DL, w))
            while self.hist and t - self.hist[0][0] > self.arm_s + 1.5 / self.fps:
                self.hist.popleft()
            ws = np.array([h[2] for h in self.hist])
            if t - self.hist[0][0] < self.arm_s - 0.5 / self.fps or ws.std() / ws.mean() >= 0.05:
                fail = "arming"
            else:
                fit = [h for h in self.hist if t - h[0] <= self.fit_s]
                vL = theil_sen([h[0] for h in fit], [h[1] for h in fit]) if len(fit) >= 8 else 0.0   # d(DL)/dt
                self.last_v = vL
                if abs(vL) > self.vmax:
                    fail = "lat_speed"
        else:
            self.hist.clear()
        if fail is not None and fail not in ("state", "arming"):
            if self.active is not None or t - self.last_active_t < 1.5:
                if t > self.suppress_until:
                    self.n_cross += 1
                self.suppress_until = t + self.cross_s
        if fail == "lat_speed":
            self.hist.clear()
        self.fail, self.armed = fail, fail is None
        self.stats["armed" if self.armed else fail] += 1
        out = dict(armed=self.armed, side=None, event=None, d=None, v=None, tlc=None)
        if not self.armed:
            for s in self.cnt:
                self.cnt[s] = 0
            if self.active is not None:                   # disarming cancels an active alert at once
                self.last_active_t = t
            self.active = None
            return out

        d_now = {"left": DL - self.car_w / 2, "right": (self.lane_w - DL) - self.car_w / 2}
        speed = {"left": -vL, "right": vL}          # approach speed toward each line (m/s)
        DL0 = self.hist[0][1]                             # sustained drift: net approach over the last ~1 s
        dnet = {"left": DL0 - DL, "right": DL - DL0}
        for s in ("left", "right"):
            d, v = d_now[s], speed[s]
            tlc = d / v if v > 1e-3 else math.inf
            cond = (d < self.margin + 0.3 * v and d > -0.5 and v >= self.vmin and tlc < 1.0 and dnet[s] >= 0.25
                    and t >= self.cool[s] and t >= self.suppress_until)
            if cond:
                self.cnt[s] += 1
                self.off[s] = 0
                if self.cnt[s] >= self.on_n:
                    if self.active != s:
                        out["event"] = s
                    self.active = s
            else:
                self.cnt[s] = 0
                if self.active == s:
                    self.off[s] += 1
                    if self.off[s] >= self.off_n:
                        self.active = None
                        self.cool[s] = t + self.cool_s
                        self.last_active_t = t
            if self.active == s:
                out.update(d=d, v=v, tlc=tlc)
        out["side"] = self.active
        if self.active is not None:
            self.last_active_t = t
        return out


# ============================================================================= Drawing
def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class TrafficLightTracker:
    """Small dedicated tracker for traffic lights: greedy matching against a constant-velocity prediction of the
    box centre and log-size (lights move smoothly away from the vanishing point while the car drives on)."""

    def __init__(self, fps, max_lost_s=0.25):
        self.max_lost = max(1, int(round(max_lost_s * fps)))
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _predict(tr, t):
        h = tr["hist"]
        if len(h) < 3:
            return h[-1]["box"]
        tt = np.array([e["t"] for e in h[-6:]])
        bb = np.array([e["box"] for e in h[-6:]])
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        lw, lh = np.log(bb[:, 2] - bb[:, 0]), np.log(bb[:, 3] - bb[:, 1])
        p = [v[-1] + theil_sen(tt, v) * (t - tt[-1]) for v in (cx, cy, lw, lh)]
        w, hh = np.exp(p[2]), np.exp(p[3])
        return [p[0] - w / 2, p[1] - hh / 2, p[0] + w / 2, p[1] + hh / 2]

    def update(self, t, dets):
        """dets: list of dict(box, conf, state). Returns the tracks closed (lost for too long) this frame."""
        pairs = []
        for k, tr in self.tracks.items():
            pb = self._predict(tr, t)
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            for d, det in enumerate(dets):
                b = det["box"]
                iou = _box_iou(pb, b)
                dc = np.hypot((b[0] + b[2]) / 2 - pc[0], (b[1] + b[3]) / 2 - pc[1])
                if (iou > 0.1 or dc < max(pw, ph)) and 0.5 < (b[3] - b[1]) / max(ph, 1e-3) < 2.0:
                    pairs.append((iou - 0.01 * dc / max(pw, ph, 1), k, d))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, k, d in pairs:
            if k in used_t or d in used_d:
                continue
            used_t.add(k)
            used_d.add(d)
            self.tracks[k]["hist"].append(dict(t=t, **dets[d]))
            self.tracks[k]["lost"] = 0
        for d, det in enumerate(dets):
            if d not in used_d:
                self.tracks[self.next_id] = dict(id=self.next_id, hist=[dict(t=t, **det)], lost=0, done=False)
                self.next_id += 1
        closed = []
        for k in list(self.tracks):
            tr = self.tracks[k]
            if len(tr["hist"]) > 120:
                del tr["hist"][:-90]                     # keep ~3 s of history
            if tr["hist"][-1]["t"] != t:                 # not updated this frame (new tracks were)
                tr["lost"] += 1
                if tr["lost"] > self.max_lost:
                    closed.append(self.tracks.pop(k))
        return closed


class RLRMonitor:
    """Red-light running: the car drives under / past a RED traffic light that belongs to its path.
    Without the vehicle speed, 'driving past' is seen in the image: a light ahead of us grows and moves up until it
    leaves the image through the TOP; a light over our own path leaves through the central part of the top edge
    (lights over other lanes / cross streets slide out through the sides). Alert when a light:
      * leaves through the top (its box touches the top edge, or it is lost inside the top band),
      * at a horizontal position inside the central zone (--rlr-zone, fractions of the width),
      * was tracked for >= 1 s, is >= 10 px tall, and was approaching (log-size growing >= 0.15/s and moving up),
      * was RED: >= 50% red (and <= 15% green/yellow) over its last 2 s, red at the moment it left,
      * had been red for >= 1.5 s (a light that only just turned red = the car entered on amber -> no alert).
    Colours are only read while the box is not cut by the top edge, ignoring the last 0.2 s before it (the red
    lamp, on top, leaves first).
    One alert per intersection (4 s cooldown)."""

    def __init__(self, W, H, fps, zone=(0.25, 0.75), top_cut=0.01, top_band=0.12, min_track_s=1.0,
                 min_growth=0.15, red_min_s=1.5, cooldown_s=4.0):
        self.W, self.H = W, H
        self.zone, self.top_cut, self.top_band = zone, top_cut, top_band
        self.min_track_s, self.min_growth = min_track_s, min_growth
        self.red_min_s, self.cooldown_s = red_min_s, cooldown_s
        self.trk = TrafficLightTracker(fps)
        self.last_alert = -1e9

    def _evaluate(self, tr):
        H, W = self.H, self.W
        full = [e for e in tr["hist"] if e["box"][1] > self.top_cut * H]       # not cut by the top edge
        if len(full) < 5:
            return None
        last = full[-1]
        b = last["box"]
        cx = (b[0] + b[2]) / 2 / W
        if not (self.zone[0] <= cx <= self.zone[1]):
            return None
        if last["t"] - full[0]["t"] < self.min_track_s or b[3] - b[1] < 10:
            return None
        win = [e for e in full if e["t"] >= last["t"] - 1.0]
        tt = [e["t"] for e in win]
        g = theil_sen(tt, [np.log(e["box"][3] - e["box"][1]) for e in win])
        vy = theil_sen(tt, [(e["box"][1] + e["box"][3]) / 2 / H for e in win])
        if g < self.min_growth or vy > -0.03:
            return None
        # colour: ignore the last 0.2 s before the light touches the top edge (the box is then often already
        # partly outside the image, so the red lamp - on top - is cut off and the remaining lamps look lit)
        col = [e for e in full if e["t"] <= last["t"] - 0.2]
        if not col:
            return None
        w2 = [e for e in col if e["t"] >= last["t"] - 2.2]
        n = len(w2)
        n_red = sum(e["state"] == "red" for e in w2)
        n_other = sum(e["state"] in ("green", "yellow") for e in w2)
        if n_red < max(6, 0.5 * n) or n_other > 0.15 * n:
            return None
        recent = [e for e in col if e["t"] >= last["t"] - 0.7 and e["state"] is not None]
        if not recent or recent[-1]["state"] != "red":
            return None
        red_since = last["t"]
        for e in reversed(col):
            if e["state"] in ("green", "yellow"):
                break
            if e["state"] == "red":
                red_since = e["t"]
        red_for = last["t"] - red_since
        if red_for < self.red_min_s:
            return None
        return dict(id=tr["id"], cx=cx, red_for=red_for, growth=g, box=b)

    def update(self, t, frame, boxes):
        """boxes: list of (xyxy, conf) traffic-light detections of this frame -> dict(event, ...)."""
        H = self.H
        dets = [dict(box=[float(v) for v in b], conf=float(c),
                     state=traffic_light_state(frame, b) if b[1] > self.top_cut * H else None) for b, c in boxes]
        closed = self.trk.update(t, dets)
        cands = [tr for tr in self.trk.tracks.values()
                 if not tr["done"] and tr["lost"] == 0 and tr["hist"][-1]["box"][1] <= self.top_cut * H]
        cands += [tr for tr in closed if not tr["done"] and tr["hist"][-1]["box"][1] < self.top_band * H]
        event = None
        for tr in cands:
            tr["done"] = True
            info = self._evaluate(tr)
            if info is not None and t - self.last_alert >= self.cooldown_s:
                self.last_alert = t
                event = info
        return dict(event=event)


def put_label(img, text, x, y, color, scale=0.6, thick=2):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - bl - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick - 1, cv2.LINE_AA)


def draw_detections(img, dets, frame_raw, lead):
    lead_id = lead[0] if lead else None
    for tid, cls, box, conf in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = BOX_COLORS.get(cls, (255, 255, 255))
        label = COCO_KEEP[cls]
        if cls == 9:
            st = traffic_light_state(frame_raw, box)
            if st:
                label += f" ({st})"
                color = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0)}[st]
        if tid is not None and tid == lead_id:
            continue  # drawn separately
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        put_label(img, f"{label} {conf:.2f}", x1, y1 - 2, color, 0.5, 2)


class AlertDisplay:
    """Visual-only persistence of the FCW / LDW alerts: keeps showing an alert for `hold_s` seconds after it was
    last active (banner, highlighted boundary, lead-box highlight). A live alert always wins, so a new or
    escalated alert (e.g. FCW critical) replaces the held banner at once. Events / CSV are not affected."""

    def __init__(self, hold_s):
        self.hold = hold_s
        self.fcw = self.ldw = None              # (t_last_active, snapshot)
        self.rlr = None                         # (t_event, event)

    def update_rlr(self, t, rlr):
        """RLR is an instantaneous event: its banner stays for max(hold, 1.5) s."""
        if rlr is not None and rlr["event"] is not None:
            self.rlr = (t, rlr["event"])
        if self.rlr is not None and t - self.rlr[0] >= max(self.hold, 1.5):
            self.rlr = None
        return self.rlr[1] if self.rlr is not None else None

    def update(self, t, fcw, ldw, lead):
        """-> (fcw_view, ldw_view, held_lead_box): dicts shaped like the monitors' outputs, extended with the
        held state; held_lead_box is the bbox to highlight while an FCW is only being held."""
        fv = ldw_v = None
        if fcw is not None and fcw["active"]:
            self.fcw = (t, dict(fcw, box=None if lead is None else lead[2]))
        if self.fcw is not None:
            t0, snap = self.fcw
            if fcw is not None and fcw["active"]:
                fv = dict(fcw)
            elif t - t0 < self.hold:
                fv = dict(snap, active=True, held=True)
            else:
                self.fcw = None
        if ldw is not None and ldw["side"] is not None:
            self.ldw = (t, dict(ldw))
        if self.ldw is not None:
            t0, snap = self.ldw
            if ldw is not None and ldw["side"] is not None:
                ldw_v = dict(ldw)
            elif t - t0 < self.hold:
                ldw_v = dict(ldw if ldw is not None else snap, side=snap["side"], armed=ldw["armed"] if ldw else None)
            else:
                self.ldw = None
        box = fv.get("box") if fv is not None and fv.get("held") else None
        return fv, ldw_v, box


def draw_fcw_box(img, box, level, frame_i):
    x1, y1, x2, y2 = [int(v) for v in box]
    crit = level == "critical"
    hc = (0, 0, 255) if crit and (frame_i // 5) % 2 == 0 else (0, 140, 255)
    cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 0), 12)
    cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), hc, 8)


def draw_lead(img, lead, fcw=None, frame_i=0):
    tid, cls, box, dist = lead
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
    txt = f"LEAD {COCO_KEEP[cls]}  {dist:.1f} m"
    if fcw is not None and fcw["ttc"] is not None and fcw["ttc"] < 10:
        txt += f"  TTC {fcw['ttc']:.1f} s"
    put_label(img, txt, x1, y1 - 4, color, 0.8, 2)
    if fcw is not None and fcw["active"]:                      # FCW: highlight the lead box
        draw_fcw_box(img, box, fcw["level"], frame_i)
    p_obj = ((x1 + x2) // 2, y2)
    p_cam = (W // 2, H - 1)
    cv2.line(img, p_obj, p_cam, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.line(img, p_obj, p_cam, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(img, p_obj, 7, (0, 255, 255), -1, cv2.LINE_AA)
    mid = ((p_obj[0] + p_cam[0]) // 2 + 12, (p_obj[1] + p_cam[1]) // 2)
    put_label(img, f"{dist:.1f} m", mid[0], mid[1], (0, 255, 255), 0.9, 2)


def draw_banner(img, text, color, y, flash=False, frame_i=0):
    """Large centred banner (the filled colour flashes when `flash`)."""
    W = img.shape[1]
    scale, thick = 1.6, 4
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x0, y0 = (W - tw) // 2 - 24, y
    fill = color if not flash or (frame_i // 5) % 2 == 0 else (40, 40, 160)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 48, y0 + th + bl + 28), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 + 4, y0 + 4), (x0 + tw + 44, y0 + th + bl + 24), fill, -1)
    cv2.putText(img, text, (x0 + 24, y0 + th + 14), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_alerts(img, fcw, ldw, frame_i, rlr=None):
    y = 70
    if rlr is not None:
        draw_banner(img, "RED LIGHT RUNNING", (0, 0, 230), y, flash=True, frame_i=frame_i)
        y += 90
    if fcw is not None and fcw["active"]:
        crit = fcw["level"] == "critical"
        txt = "FCW CRITICAL" if crit else "FCW WARNING"
        if fcw["ttc"] is not None:
            txt += f"  TTC {fcw['ttc']:.1f} s"
        draw_banner(img, txt, (0, 0, 230) if crit else (0, 140, 255), y, flash=crit, frame_i=frame_i)
        y += 90
    if ldw is not None and ldw["side"] is not None:
        txt = "LDW << LEFT" if ldw["side"] == "left" else "LDW RIGHT >>"
        draw_banner(img, txt, (0, 110, 255), y)


# ============================================================================= Main
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
PROFILES = {
    # best quality (default). Inference alone on an RTX 4070 Laptop: YOLO ~27 fps, SegFormer ~13 fps
    "accurate": dict(seg_model=os.path.join(MODELS_DIR, "segformer-b5-cityscapes"), seg_size="1280x720",
                     det_model=os.path.join(MODELS_DIR, "yolo26x.pt"), det_imgsz=1280),
    # real-time. Inference alone: YOLO ~79 fps, SegFormer ~81 fps
    "fast": dict(seg_model=os.path.join(MODELS_DIR, "segformer-b1-cityscapes"), seg_size="1024x576",
                 det_model=os.path.join(MODELS_DIR, "yolo26s.pt"), det_imgsz=960),
}


class StageTimer:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.tot = defaultdict(float)
        self.n = defaultdict(int)

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                if timer.use_cuda:
                    torch.cuda.synchronize()
                timer.tot[name] += time.perf_counter() - self.t
                timer.n[name] += 1
        return _Ctx()

    def report(self):
        parts = []
        for k in self.tot:
            ms = 1000 * self.tot[k] / max(self.n[k], 1)
            parts.append(f"{k} {ms:.1f} ms ({1000 / ms:.0f} fps)" if ms > 0 else f"{k} 0 ms")
        return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser(
        prog="dashcam_vision.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Computer vision for dashcam video:
  * Cityscapes semantic segmentation (19 classes) with SegFormer (NVIDIA).
  * Detection + tracking of people, vehicles and traffic lights with YOLO26 (COCO),
    including the traffic-light colour (red/yellow/green).
  * Lead-vehicle selection (in the ego lane) and distance estimate, drawn as a
    line from the bottom of its bbox to the camera.
  * ADAS alerts: FCW (forward collision, from TTC), LDW (lane departure) and RLR (red-light running).

Outputs (PREFIX = -o PREFIX; default outputs/<input video name>):
  PREFIX_segmentation.mp4   segmentation only, with legend
  PREFIX_detections.mp4     detections only + lead vehicle and distance
  PREFIX_combined.mp4       both together
  PREFIX_alerts.csv         FCW / LDW / RLR events (time, frame, type, level/side, TTC, distance...);
                            written together with the videos (not with --show)""",
        epilog="""\
profiles (--profile):
  accurate  SegFormer-B5 @1280x720 + YOLO26x @1280  -> best quality
            (inference only, RTX 4070 Laptop: seg ~13 fps, det ~28 fps)
  fast      SegFormer-B1 @1024x576 + YOLO26s @960   -> real time
            (inference only, RTX 4070 Laptop: seg ~70 fps, det ~80 fps)
  --seg-model, --seg-size, --det-model and --det-imgsz override the profile.

distance estimation (approximate; the camera is not calibrated):
  width cue  : Z = f * real_width / bbox_width_px
  ground cue : Z = f * camera_height / (bbox_bottom_y - horizon_y)
  Both are fused and smoothed over time. f comes from --hfov.
  If you know the real camera setup, tune --hfov, --cam-height and --horizon.

--show mode:
  With --show nothing is written: live window only. Without --show the videos are written
  (and the measured fps is closest to the real one, without the imshow cost).
  Keys: q / ESC quits, SPACE pauses / resumes.

examples:
  # whole video, heavy models, writes the 3 videos (no -o: outputs/sp1_crop_*.mp4)
  %(prog)s sp1_crop.mp4
  # light models (real time)
  %(prog)s sp1_crop.mp4 -o outputs/sp1_fast --profile fast
  # detection only, segmentation only
  %(prog)s sp1_crop.mp4 -o outputs/sp1_det --profile fast --no-seg
  %(prog)s sp1_crop.mp4 -o outputs/sp1_seg --profile fast --no-det
  # live window, nothing written
  %(prog)s sp1_crop.mp4 --profile fast --show
  # quick test: 10 s starting at 2:00
  %(prog)s sp1_crop.mp4 -o outputs/test --start 120 --duration 10
  # no lane detection (back to the fixed trapezoid corridor)
  %(prog)s sp1_crop.mp4 -o outputs/sp1_nolanes --profile fast --no-lane
  # lanes only drawn, old trapezoid corridor; hold the lane for 4 s without detection
  %(prog)s sp1_crop.mp4 -o outputs/sp1_lanes --profile fast --no-lane-corridor --lanes-hold 4
  # no ADAS alerts; FCW only; more sensitive LDW; earlier FCW; alerts shown for 3 s
  %(prog)s sp1_crop.mp4 -o outputs/sp1_noalerts --profile fast --no-fcw --no-ldw
  %(prog)s sp1_crop.mp4 -o outputs/sp1_fcw --profile fast --no-ldw
  %(prog)s sp1_crop.mp4 -o outputs/sp1_ldw --profile fast --ldw-margin 0.4 --fcw-ttc 3.5 --alert-display 3
  # no red-light-running alert; wider central exit zone for RLR
  %(prog)s sp1_crop.mp4 -o outputs/sp1_norlr --profile fast --no-rlr
  %(prog)s red_light.mp4 -o outputs/red_light --rlr-zone 0.2 0.8

lanes (on by default; --no-lane turns them off):
  Polar R-CNN finds the lanes; the tracker picks the left/right boundary of the ego lane
  (plausible width ~600-1300 px at the bottom row, = 2.7-4.0 m with the camera at 1.5 m),
  smooths it over time and rejects jumps. State shown on the video: detected / partial (one
  boundary inferred from the previous width) / estimated (nothing seen: the last lane is
  held, dashed and fading, for --lanes-hold s; assumes driving straight) / none (falls back
  to the trapezoid corridor).

ADAS alerts (on by default; --no-fcw / --no-ldw / --no-rlr turn them off):
  FCW (forward collision): TTC = 1 / (d ln(bbox_width) / dt), robust fit (Theil-Sen) over 0.8 s
    on the lead vehicle; needs no calibration. Only valid with the same vehicle for >= 0.5 s,
    bbox >= 40 px wide, distance < 40 m, bbox fully inside the image, inside the ego lane (no
    margin), bbox height growing consistently with the width, implied closing speed <= 14 m/s
    and the smoothed distance also shrinking (>= 0.5 m/s). 'warning' when TTC < --fcw-ttc,
    'critical' when TTC < --fcw-ttc-critical; needs 0.3 s in a row to trigger and 0.5 s
    without the condition to release. Requires detection.
  LDW (lane departure): the priority is NO false alarms (missing a borderline case is fine).
    Lateral position = image centre (camera assumed centred on the car) relative to the two
    lane boundaries, lane width assumed 3.5 m; wheel-to-line distance = boundary distance -
    half of --ldw-car-width. Fires when the wheel is within --ldw-margin (plus a speed-dependent
    allowance) of the line AND the car is approaching it (>= 0.15 m/s, time to cross < 1 s, net
    drift >= 0.25 m over the last 1 s) for 0.3 s; then 2.5 s without repeating on that side.
    LDW is only ARMED if, for >= 1 s in a row, BOTH boundaries were actually observed (state
    'detected', never partial/estimated), with no tracker jumps or re-acquisitions, with a
    plausible and stable width, straight boundaries converging at the horizon and consistent
    with the frame observation. Any failure disarms at once. When the line is actually crossed
    (the boundaries swap) LDW is suppressed for 2 s more. Requires lanes: it is turned off with
    --no-lane / --no-det.
    Limitation: without the car speed (no CAN/GPS) LDW cannot be suppressed at low speed
    (parking manoeuvres, stopped traffic) nor know whether the turn signal is on.
  RLR (red-light running): without the car speed, 'driving past the light' is read from the image:
    a traffic light ahead grows and moves up until it leaves the image through the TOP edge. A light
    over our own path leaves through the central part of the top edge; lights over other lanes or
    cross streets slide out through the sides, so only lights leaving inside --rlr-zone (default:
    the central half of the width) count. Fires when such a light, tracked for >= 1 s and visibly
    approaching (growing >= 15%%/s, moving up), was red for >= 50%% of its last 2 s, still red when
    it left and red for >= 1.5 s (a light that only just turned red = the car entered on amber ->
    no alert). The lamp colour comes from the position of the lit lamp in the housing and its colour
    relative to the other lamps. One alert per intersection (4 s). Stopping at a red light gives no
    alert (the light does not grow). Requires detection.
    Limitations: lights on a pole at the side of the road (common in Brazil) leave through the side
    and are not checked; at night YOLO often sees only the lamp glow, not the housing, so lights
    are detected much less (night RLR is mostly missed).
  Events are written to PREFIX_alerts.csv.
  --alert-display only affects how long banners / highlights stay on screen, not the events.""")

    g = ap.add_argument_group("help")
    g.add_argument("-h", "--help", action="help", help="show this help and exit")

    g = ap.add_argument_group("input and output")
    g.add_argument("input", help="input video (e.g. sp1_crop.mp4)")
    g.add_argument("-o", "--out-prefix", default=None, metavar="PREFIX",
                   help="prefix of the output videos; writes PREFIX_segmentation.mp4, PREFIX_detections.mp4 and "
                        "PREFIX_combined.mp4 (default: outputs/<input video name without extension>; the folder is "
                        "created if missing)")
    g.add_argument("--start", type=float, default=0.0, metavar="SEC",
                   help="start processing at this time, in seconds (default: 0)")
    g.add_argument("--duration", type=float, default=None, metavar="SEC",
                   help="process only this many seconds (default: until the end of the video)")
    g.add_argument("--no-combined", action="store_true",
                   help="do not write the combined video (only separate segmentation and detections)")
    g.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto",
                   help="output video encoder: nvenc (on the GPU, fast), x264 (on the CPU, slightly better "
                        "per MB) or auto = nvenc if available (default: auto)")
    g.add_argument("--no-audio", action="store_true", help="do not copy the audio of the source video to the outputs")

    g = ap.add_argument_group("what to run")
    g.add_argument("--profile", choices=list(PROFILES), default="accurate",
                   help="model set: accurate (heavy, best quality) or fast (light, real time); "
                        "see 'profiles' below (default: accurate)")
    g.add_argument("--no-seg", action="store_true", help="turn segmentation off (run detection only)")
    g.add_argument("--no-det", action="store_true", help="turn detection off (run segmentation only)")
    g.add_argument("--no-lane", action="store_true",
                   help="turn lane detection (Polar R-CNN) off; the lead vehicle goes back to being chosen "
                        "by the fixed trapezoid corridor")

    g = ap.add_argument_group("live view")
    g.add_argument("--show", nargs="?", const="det", default=None, choices=["det", "seg", "comb"],
                   help="open a window showing the result while processing: det (detections, default if "
                        "only --show is given), seg (segmentation) or comb (both). With --show NO video is "
                        "written (-o, --encoder etc. are ignored)")
    g.add_argument("--show-width", type=int, default=1280, metavar="PX",
                   help="width of the --show window, in pixels (default: 1280)")

    g = ap.add_argument_group("segmentation (SegFormer, Cityscapes)")
    g.add_argument("--seg-model", default=None, metavar="HF_ID",
                   help="HuggingFace model (profile default: nvidia/segformer-b5-finetuned-cityscapes-1024-1024 "
                        "or ...-b1-...); b0, b2, b3, b4 also work")
    g.add_argument("--seg-size", default=None, metavar="WxH",
                   help="resolution the network runs at, e.g. 1280x720; smaller = faster and less detail "
                        "(profile default: 1280x720 or 1024x576)")
    g.add_argument("--seg-every", type=int, default=1, metavar="N",
                   help="run segmentation every N frames and repeat the last mask in between "
                        "(default: 1 = every frame)")
    g.add_argument("--seg-alpha", type=float, default=0.45, metavar="A",
                   help="opacity of the segmentation colours over the video, from 0 to 1 (default: 0.45)")
    g.add_argument("--no-legend", action="store_true", help="do not draw the segmentation class legend")

    g = ap.add_argument_group("detection (YOLO26, COCO)")
    g.add_argument("--det-model", default=None, metavar="WEIGHTS",
                   help="YOLO weights; downloaded automatically if missing: yolo26n/s/m/l/x.pt "
                        "(profile default: yolo26x.pt or yolo26s.pt)")
    g.add_argument("--det-imgsz", type=int, default=None, metavar="PX",
                   help="YOLO input image size; larger catches small/distant objects but is "
                        "slower (profile default: 1280 or 960)")
    g.add_argument("--conf", type=float, default=0.30, metavar="C",
                   help="minimum confidence to accept a detection, from 0 to 1 (default: 0.30)")
    g.add_argument("--tracker", default="bytetrack.yaml", metavar="YAML",
                   help="Ultralytics tracker: bytetrack.yaml or botsort.yaml (default: bytetrack.yaml)")

    g = ap.add_argument_group("lead vehicle and distance")
    g.add_argument("--hfov", type=float, default=75.0, metavar="DEG",
                   help="horizontal field of view of the camera, in degrees; defines the focal length "
                        "(default: 75)")
    g.add_argument("--cam-height", type=float, default=1.5, metavar="M",
                   help="camera height above the ground, in metres (default: 1.5)")
    g.add_argument("--horizon", type=float, default=0.59, metavar="FRAC",
                   help="height of the horizon line in the image, as a fraction of the height (0 = top, "
                        "1 = bottom) (default: 0.59)")
    g.add_argument("--corridor-bottom", type=float, default=0.12, metavar="FRAC",
                   help="half-width of the lane corridor at the bottom of the image, as a fraction of the width; "
                        "a vehicle is only a 'lead' candidate if the bottom-centre of its bbox is inside "
                        "this corridor. With lanes on (default) this trapezoid corridor is only used "
                        "when there is no ego lane, with --no-lane or with --no-lane-corridor (default: 0.12)")
    g.add_argument("--corridor-top", type=float, default=0.02, metavar="FRAC",
                   help="half-width of the lane corridor at the horizon line, as a fraction of the width "
                        "(default: 0.02)")

    g = ap.add_argument_group("lanes (Polar R-CNN, CurveLanes; on by default, --no-lane turns off)")
    g.add_argument("--lanes", action="store_true", help=argparse.SUPPRESS)   # old opt-in; now the default
    g.add_argument("--lanes-thr", type=float, default=LANES_THR_DEFAULT, metavar="C",
                   help=f"minimum confidence of a detected lane, from 0 to 1 (default: {LANES_THR_DEFAULT})")
    g.add_argument("--lanes-hold", type=float, default=2.5, metavar="SEC",
                   help="for how many seconds the ego lane keeps being extrapolated (held, fading) "
                        "after it stops being detected, e.g. at an intersection (default: 2.5)")
    g.add_argument("--no-lane-corridor", action="store_true",
                   help="draw the lanes but keep the fixed trapezoid corridor (--corridor-*) to choose "
                        "the lead vehicle")

    g = ap.add_argument_group("ADAS alerts (FCW / LDW / RLR; on by default)")
    g.add_argument("--no-fcw", action="store_true", help="turn the forward collision warning (FCW) off")
    g.add_argument("--no-ldw", action="store_true",
                   help="turn the lane departure warning (LDW) off; it is also off with --no-lane / --no-det")
    g.add_argument("--no-rlr", action="store_true",
                   help="turn the red-light running (RLR) alert off; it is also off with --no-det")
    g.add_argument("--fcw-ttc", type=float, default=2.7, metavar="SEC",
                   help="FCW 'warning' level: TTC (time to collision) below this value (default: 2.7)")
    g.add_argument("--fcw-ttc-critical", type=float, default=1.5, metavar="SEC",
                   help="FCW 'critical' level (flashing red banner): TTC below this value (default: 1.5)")
    g.add_argument("--ldw-margin", type=float, default=0.25, metavar="M",
                   help="LDW fires when the wheel is within M metres of the line (plus a small allowance that "
                        "grows with the approach speed) and the car is approaching it; larger = more sensitive, "
                        "more risk of false alarms (default: 0.25)")
    g.add_argument("--ldw-car-width", type=float, default=1.8, metavar="M",
                   help="car width, in metres (the wheel is half of that from the centre) (default: 1.8)")
    g.add_argument("--rlr-zone", type=float, nargs=2, default=(0.25, 0.75), metavar=("L", "R"),
                   help="RLR: a red light only counts if it leaves the image through the top edge between "
                        "these horizontal positions, as fractions of the width (default: 0.25 0.75)")
    g.add_argument("--alert-display", type=float, default=2.0, metavar="SEC",
                   help="minimum time the FCW / LDW / RLR banners, the highlighted lane boundary and the lead-box "
                        "highlight stay on screen after the alert was last active; visual only, events and CSV "
                        "are unchanged (default: 2.0; 0 = only while active)")
    g.add_argument("--adas-debug", default=None, metavar="CSV", help=argparse.SUPPRESS)   # per-frame LDW/FCW trace
    g.add_argument("--dry", action="store_true", help=argparse.SUPPRESS)                    # no video output (tuning)
    args = ap.parse_args()
    for k, v in PROFILES[args.profile].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.no_seg and args.no_det:
        ap.error("--no-seg and --no-det together leave nothing to do")
    # lanes are on by default; they are only drawn in detections/combined, so --no-det also turns them off
    args.lanes = not (args.no_lane or args.no_det)
    args.fcw = not (args.no_fcw or args.no_det)
    args.ldw = not (args.no_ldw or not args.lanes)
    args.rlr = not (args.no_rlr or args.no_det)
    if not 0 <= args.rlr_zone[0] < args.rlr_zone[1] <= 1:
        ap.error("--rlr-zone needs 0 <= L < R <= 1")
    if args.show and ((args.show == "seg" and args.no_seg) or (args.show == "det" and args.no_det)
                      or (args.show == "comb" and (args.no_seg or args.no_det))):
        args.show = "seg" if args.no_det else "det"     # fall back to what is actually computed
    if args.show and args.out_prefix:
        print("[info] --show: no video will be written (-o ignored)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"[info] profile '{args.profile}': "
          + ("" if args.no_seg else f"seg={args.seg_model} @ {args.seg_size}  ")
          + ("" if args.no_det else f"det={args.det_model} @ {args.det_imgsz}")
          + (f"  lanes=Polar R-CNN DLA34 (CurveLanes)" if args.lanes else "  lanes=off"))
    if args.lanes:
        print("[info] lanes: drawn in the detections and combined videos (there is no lanes-only video); "
              "lead-vehicle corridor = "
              + ("fixed trapezoid (--no-lane-corridor)" if args.no_lane_corridor else "ego lane"))

    print("[info] alerts: FCW " + (f"on (TTC {args.fcw_ttc:g} s / {args.fcw_ttc_critical:g} s)" if args.fcw else "off")
          + ", LDW " + (f"on (margin {args.ldw_margin:g} m, car {args.ldw_car_width:g} m)" if args.ldw else "off")
          + ", RLR " + (f"on (zone {args.rlr_zone[0]:g}-{args.rlr_zone[1]:g} of the width)" if args.rlr else "off")
          + ("  [no vehicle speed: LDW is not suppressed at low speed]" if args.ldw else ""))

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(args.start * fps)
    end_f = total if args.duration is None else min(total, start_f + int(args.duration * fps))
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    n_frames = end_f - start_f
    print(f"[info] video {W}x{H} @ {fps:.2f} fps, processing frames {start_f}..{end_f} ({n_frames})")

    # models
    det = seg = lane_det = lane_trk = None
    lane_list, lane_corridor = [], False
    if not args.no_det:
        from ultralytics import YOLO
        det = YOLO(args.det_model)
    if not args.no_seg:
        sw, sh = (int(v) for v in args.seg_size.lower().split("x"))
        seg = Segmenter(args.seg_model, device, sw, sh)
    if args.lanes:
        from lanes_polar import PolarLaneDetector       # lazy: the lane model is not loaded with --no-lane
        lane_det = PolarLaneDetector(thr=args.lanes_thr, device=device)
        lane_trk = EgoLaneTracker(W, H, fps, args.cam_height, args.horizon, args.lanes_hold)
        lane_corridor = not args.no_lane_corridor
    fcw_mon = FCWMonitor(W, H, fps, args.fcw_ttc, args.fcw_ttc_critical) if args.fcw else None
    ldw_mon = LDWMonitor(W, H, fps, lane_trk, args.ldw_car_width, args.ldw_margin,
                         horizon_frac=args.horizon) if args.ldw else None
    rlr_mon = RLRMonitor(W, H, fps, zone=tuple(args.rlr_zone)) if args.rlr else None
    lead_est = LeadVehicleEstimator(W, H, args.hfov, args.cam_height, args.horizon,
                                    args.corridor_bottom, args.corridor_top, hold_frames=int(fps * 0.3))

    # outputs: pipe raw frames into ffmpeg (H.264), copy audio from the source segment
    prefix = args.out_prefix or os.path.join("outputs", os.path.splitext(os.path.basename(args.input))[0])
    if not args.show and not args.dry and os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    encoder = args.encoder
    if encoder == "auto":
        encs = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        encoder = "nvenc" if (device.type == "cuda" and "h264_nvenc" in encs) else "x264"
    venc = (["-c:v", "h264_nvenc", "-preset", "medium", "-rc", "vbr", "-cq", "21", "-b:v", "0"] if encoder == "nvenc"
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "20"])

    def open_writer(path):
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
        if not args.no_audio:
            cmd += ["-ss", f"{start_f / fps}", "-t", f"{n_frames / fps}", "-i", args.input,
                    "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
        cmd += venc + ["-pix_fmt", "yuv420p", path]
        print(f"[info] writing {path} ({encoder})")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    csv_f = None
    if not args.show and not args.dry and (fcw_mon or ldw_mon or rlr_mon):
        csv_path = f"{prefix}_alerts.csv"
        print(f"[info] writing {csv_path}")
        csv_f = open(csv_path, "w")
        csv_f.write("time_s,frame,type,level_side,ttc_s,line_dist_m,lat_vel_mps,tlc_s,lead_id,lead_dist_m,red_for_s\n")
    n_ev = defaultdict(int)

    def log_event(t, frame_i, kind, detail, ttc="", d="", v="", tlc="", lead=None, red_for=""):
        n_ev[kind] += 1
        f = lambda x, n=2: "" if x == "" or x is None else f"{x:.{n}f}"
        row = (f"{t:.2f},{frame_i},{kind},{detail},{f(ttc)},{f(d)},{f(v)},{f(tlc)},"
               f"{'' if lead is None else int(lead[0])},{'' if lead is None else f'{lead[3]:.1f}'},{f(red_for, 1)}")
        if csv_f:
            csv_f.write(row + "\n")
        print(f"[alert] t={t:.2f}s {kind} {detail} {row.split(',', 4)[4]}")

    dbg = open(args.adas_debug, "w") if args.adas_debug else None
    if dbg:
        dbg.write("frame,t,state,fail,w,DL,ttc,lead_id,lead_dist,lead_w,vL,y0,wres,resl,resr,sll,slr\n")
    writers = {}
    if not args.show and not args.dry:
        if det is not None:
            writers["det"] = open_writer(f"{prefix}_detections.mp4")
        if seg is not None:
            writers["seg"] = open_writer(f"{prefix}_segmentation.mp4")
        if det is not None and seg is not None and not args.no_combined:
            writers["comb"] = open_writer(f"{prefix}_combined.mp4")

    win = "dashcam_vision"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.show_width, int(args.show_width * H / W))
    live_fps, t_prev = None, time.perf_counter()

    timer = StageTimer(device.type == "cuda")
    seg_map, dets, lead, t0 = None, [], None, time.time()
    fcw = ldw = rlr = None
    alert_disp = AlertDisplay(args.alert_display)
    for i in range(n_frames):
        with timer("decode"):
            ok, frame = cap.read()
        if not ok:
            break

        # --- lanes (Polar R-CNN) + ego-lane tracking
        if lane_det is not None:
            with timer("lanes"):
                lane_list, _ = lane_det(frame)
                lane_trk.update(lane_list)

        # --- detection + tracking
        if det is not None:
            with timer("det"):
                res = det.track(frame, persist=True, tracker=args.tracker, conf=args.conf, imgsz=args.det_imgsz,
                                classes=list(COCO_KEEP), quantize=16 if device.type == "cuda" else None,
                                verbose=False)[0]
                dets = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    conf = res.boxes.conf.cpu().numpy()
                    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(cls)
                    dets = [(ids[k], cls[k], xyxy[k], conf[k]) for k in range(len(cls))]
                lead = lead_est.update([(t, c, b) for t, c, b, _ in dets if t is not None],
                                       lane_trk if lane_corridor else None)

        # --- ADAS alerts (FCW needs the lead vehicle, LDW the ego-lane tracker)
        if fcw_mon is not None or ldw_mon is not None or rlr_mon is not None:
            with timer("adas"):
                t_s = (start_f + i) / fps
                if fcw_mon is not None:
                    lead_f = lead                        # FCW: the lead must be inside the ego lane (no tolerance margin)
                    if lead is not None and lane_trk is not None and lane_corridor and lane_trk.available \
                            and not lane_trk.contains(lead[2], margin=0.0):
                        lead_f = None
                    fcw = fcw_mon.update(t_s, lead_f)
                    if fcw["event"]:
                        log_event(t_s, start_f + i, "FCW", fcw["event"], ttc=fcw["ttc"], lead=lead)
                if ldw_mon is not None:
                    ldw = ldw_mon.update(t_s)
                    if ldw["event"]:
                        log_event(t_s, start_f + i, "LDW", ldw["event"], d=ldw["d"], v=ldw["v"], tlc=ldw["tlc"], lead=lead)
                if rlr_mon is not None:
                    rlr = rlr_mon.update(t_s, frame, [(b, c) for _, cl, b, c in dets if cl == 9])
                    if rlr["event"]:
                        log_event(t_s, start_f + i, "RLR", "red", red_for=rlr["event"]["red_for"])

        if dbg:
            r = ldw_mon.row if ldw_mon else 0
            ok = lane_trk is not None and lane_trk.xl is not None
            w_ = lane_trk.xr[r] - lane_trk.xl[r] if ok else float("nan")
            dl = (W / 2 - lane_trk.xl[r]) / w_ * 3.5 if ok else float("nan")
            dbg.write(f"{start_f + i},{(start_f + i) / fps:.3f},{lane_trk.state if lane_trk else ''},"
                      f"{ldw_mon.fail if ldw_mon else ''},{w_:.1f},{dl:.3f},"
                      f"{'' if not fcw or fcw['ttc'] is None else round(fcw['ttc'], 2)},"
                      f"{'' if lead is None else int(lead[0])},{'' if lead is None else round(lead[3], 1)},"
                      f"{'' if lead is None else round(lead[2][2] - lead[2][0])},"
                      f"{'' if not ldw_mon or ldw_mon.last_v is None else round(ldw_mon.last_v, 3)},"
                      + ",".join(f"{ldw_mon.m.get(k, float('nan')):.3f}" for k in ("y0", "wres", "resl", "resr", "sll", "slr")) + "\n")

        # --- segmentation
        if seg is not None and (seg_map is None or i % args.seg_every == 0):
            with timer("seg"):
                seg_map = seg(frame)

        # alerts as displayed (held for --alert-display s after the last active frame)
        fcw_v, ldw_v, held_box = alert_disp.update((start_f + i) / fps, fcw, ldw, lead)
        rlr_v = alert_disp.update_rlr((start_f + i) / fps, rlr)

        def draw_dets_on(img):
            if lane_det is not None:
                draw_lanes(img, lane_list, lane_trk, ldw_v["side"] if ldw_v else None, ldw["armed"] if ldw else None)
            draw_detections(img, dets, frame, lead)
            if lead is not None:
                draw_lead(img, lead, None if (fcw_v and fcw_v.get("held")) else fcw_v, i)
            if held_box is not None:                  # held FCW highlight (live alert already drawn by draw_lead)
                draw_fcw_box(img, held_box, fcw_v["level"], i)
            draw_alerts(img, fcw_v, ldw_v, i, rlr_v)
            return img

        with timer("draw+write"):
            imgs = {}
            if det is not None:                       # 1) detections only (on the raw frame)
                imgs["det"] = draw_dets_on(frame.copy())
            if seg is not None:                       # 2) segmentation only
                imgs["seg"], legend_ids = seg.overlay(frame, seg_map, args.seg_alpha)
                if not args.no_legend:
                    draw_legend(imgs["seg"], legend_ids)
                if det is not None and ("comb" in writers or args.show == "comb"):   # 3) combined
                    imgs["comb"] = draw_dets_on(imgs["seg"].copy())
            for k, w in writers.items():
                w.stdin.write(imgs[k].tobytes())

        # --- live window
        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        live_fps = inst if live_fps is None else 0.9 * live_fps + 0.1 * inst
        t_prev = now
        if args.show:
            with timer("show"):
                view = cv2.resize(imgs[args.show], (args.show_width, int(args.show_width * H / W)),
                                  interpolation=cv2.INTER_LINEAR)
                put_label(view, f"{live_fps:.1f} fps  [{args.profile}]", view.shape[1] - 230, 28,
                          (255, 255, 255), 0.6, 2)
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    key = cv2.waitKey(0) & 0xFF
                closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), 27) or closed:
                print("[info] stopped by user")
                break

        if (i + 1) % 100 == 0 or i + 1 == n_frames:
            el = time.time() - t0
            fps_run = (i + 1) / el
            print(f"[{i + 1}/{n_frames}] total {fps_run:.1f} fps, ETA {(n_frames - i - 1) / fps_run / 60:.1f} min"
                  f"  ||  {timer.report()}", flush=True)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    for w in writers.values():
        w.stdin.close()
        w.wait()
    if csv_f:
        csv_f.close()
    if dbg:
        dbg.close()
    if fcw_mon or ldw_mon or rlr_mon:
        msg = f"[info] alerts: FCW {n_ev['FCW']}, LDW {n_ev['LDW']}, RLR {n_ev['RLR']}"
        if ldw_mon:
            tot = max(sum(ldw_mon.stats.values()), 1)
            msg += (f" | LDW armed in {100 * ldw_mon.stats['armed'] / tot:.0f}% of frames, crossings suppressed "
                    f"{ldw_mon.n_cross}; disarm reasons: " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                                                              sorted(ldw_mon.stats.items(), key=lambda kv: -kv[1]) if k != "armed"))
        print(msg)
    print("[done]")


if __name__ == "__main__":
    main()
