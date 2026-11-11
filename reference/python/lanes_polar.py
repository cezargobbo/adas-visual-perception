#!/usr/bin/env python
"""Polar R-CNN (CurveLanes, DLA34, NMS-free): model + importable lane detector.

Pure-PyTorch reimplementation of the inference path of the ShqWW/PolarRCNN repo
(without the NMSCUDA CUDA op: the NMS-free mode in the config does not use NMS), with parameter
names identical to the checkpoint (load_state_dict strict=True). The DLA34 backbone is imported
from the repo itself (third_party/PolarRCNN/Models/Backbone/dla34.py, pure PyTorch, pretrained=False).
Config: Config/polarrcnn_curvelanes_dla34.py
"""
import importlib.util, math, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- config polarrcnn_curvelanes_dla34.py ----
ORI_H, ORI_W = 1440, 2560
CUT_FRAC = 640 / 1440                         # crop 640 of 1440 rows from the top (CurveLanes convention; 480 at 1080p)
CUT_H = 640
IMG_H, IMG_W = 320, 800
CENTER_H, CENTER_W = 15, 386
NUM_OFFSETS = 72
POLAR_H, POLAR_W = 6, 13
NUM_TEST_PRIORS = 50
NEG_FEAT = 36
FC_DIM = 192
PRIOR_CH = 64
NUM_GROUPS = 6
GNN_DIM = 128
IOU_DIM = 5
O2O_ANGLE, O2O_RHO = math.pi / 6, 50
CONF_THRES = 0.45                             # conf_thres (internal filter of the o2o head)
CONF_NMSFREE = 0.44                           # conf_thres_nmsfree (final threshold, NMS-free)
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 128, 128), (255, 128, 128),
          (0, 255, 128), (128, 255, 0), (200, 200, 200), (60, 120, 255)]


def load_dla():
    path = os.path.join(HERE, 'third_party/PolarRCNN/Models/Backbone/dla34.py')
    spec = importlib.util.spec_from_file_location('polar_dla34', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.DLAWrapper(pretrained=False)


# ------------------------------------------------------------------ model
class FPN(nn.Module):
    def __init__(self, cins=(128, 256, 512), c=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(i, c, 1) for i in cins])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(c, c, 3, 1, 1) for _ in cins])

    def forward(self, xs):
        lat = [l(x) for l, x in zip(self.lateral_convs, xs)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode='nearest')
        return [c(l) for c, l in zip(self.fpn_convs, lat)]


def img2car(p):
    q = torch.zeros_like(p); q[..., 0] = p[..., 0] - CENTER_W; q[..., 1] = CENTER_H - p[..., 1]; return q


def car2img(p):
    q = torch.zeros_like(p); q[..., 0] = p[..., 0] + CENTER_W; q[..., 1] = CENTER_H - p[..., 1]; return q


class LocalPolarHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_layers = nn.Conv2d(64, 2, 1, 1, 0, bias=False)
        self.cls_layers = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
        gy, gx = torch.meshgrid(torch.arange(0.5, POLAR_H, 1, dtype=torch.float32),
                                torch.arange(0.5, POLAR_W, 1, dtype=torch.float32), indexing='ij')
        grid = torch.stack((gx * IMG_W / POLAR_W, gy * IMG_H / POLAR_H), -1)
        self.register_buffer('grid_car', img2car(grid))
        self.register_buffer('prior_id', torch.linspace(0, POLAR_H * POLAR_W - 1, POLAR_H * POLAR_W, dtype=torch.int32))

    def forward(self, feats):
        x = F.interpolate(feats[-1], size=(POLAR_H, POLAR_W), mode='bilinear', align_corners=True)
        reg = torch.arctan(self.reg_layers(x)) / math.pi
        cls = self.cls_layers(x).sigmoid().squeeze(1)
        angle = reg[:, 0] * math.pi
        local_rho = reg[:, 1] * IMG_W / POLAR_W
        g_rho = local_rho + self.grid_car[..., 0] * torch.cos(angle) + self.grid_car[..., 1] * torch.sin(angle)
        angle = (angle / math.pi).clamp(-0.45, 0.45)
        emb = torch.stack((angle, g_rho / IMG_W), -1).flatten(1, 2)            # (B,78,2)
        ids = self.prior_id.unsqueeze(0).expand(emb.shape[0], -1)
        top = torch.topk(cls.flatten(1), k=NUM_TEST_PRIORS, dim=-1, largest=True, sorted=False)[1]
        emb = torch.gather(emb, 1, top[..., None].expand(-1, -1, 2))
        ids = torch.gather(ids, 1, top)
        return emb, ids


class LaneGNN(nn.Module):
    def __init__(self, d=FC_DIM, inter=GNN_DIM, iou=IOU_DIM):
        super().__init__()
        self.pos_linear = nn.Linear(2, inter)
        self.linear_in = nn.Linear(d, inter)
        self.linear_out = nn.Linear(d, inter)
        self.edge_layers = nn.Sequential(nn.Linear(inter, inter), nn.GELU(), nn.Linear(inter, iou))
        self.node_layers = nn.Sequential(nn.Linear(iou, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())

    def forward(self, nf, supress, pos_mat):
        e = self.linear_in(nf).unsqueeze(2) - self.linear_out(nf).unsqueeze(1) + self.pos_linear(pos_mat)
        e = self.edge_layers(e) * supress.unsqueeze(-1)
        return self.node_layers(e.max(dim=1)[0])


class O2OClsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_block = nn.Sequential(nn.Linear(FC_DIM, FC_DIM), nn.ReLU())
        self.lane_gnn = LaneGNN()
        self.cls_layer = nn.Linear(FC_DIM, 1)
        y = CENTER_H - torch.linspace(0, 1 - 1e-5, steps=NUM_OFFSETS, dtype=torch.float32) * IMG_H
        self.register_buffer('sample_car_y', y[0:2])

    def nbr_mask(self, emb):
        angle = emb[..., 0] * math.pi
        ad = (angle.unsqueeze(-1) - angle.unsqueeze(-2)).abs()
        ad = torch.triu(ad, diagonal=1); ad = ad + ad.transpose(-1, -2)
        # faithful to the repo: the rho matrix is (mistakenly) derived from the angle matrix
        rd = torch.triu(ad, diagonal=1); rd = rd + rd.transpose(-1, -2)
        return (ad < O2O_ANGLE) & (rd < O2O_RHO)

    def sample_point(self, emb):
        angle = emb[..., 0] * math.pi
        rho = emb[..., 1] * IMG_W
        cy = self.sample_car_y.view(1, 1, 2)
        cx = -torch.tan(angle).unsqueeze(-1) * cy + (rho / torch.cos(angle)).unsqueeze(-1)
        return cx / IMG_W

    def forward(self, feats, cls_pred, ids, emb):
        nbr = self.nbr_mask(emb)
        cm = cls_pred.unsqueeze(-1) - cls_pred.unsqueeze(-2)
        im = ids.unsqueeze(-1) - ids.unsqueeze(-2)
        sup = ((cm > 0) | ((cm == 0) & (im > 0))).float()
        sup[nbr == 0] = 0
        pos = self.sample_point(emb)
        pos_mat = pos.unsqueeze(2) - pos.unsqueeze(1)
        f = self.lane_gnn(self.cls_block(feats), sup, pos_mat)
        logits = self.cls_layer(f).squeeze(-1)
        logits = logits.masked_fill(cls_pred < CONF_THRES, -1e6)
        return logits.sigmoid()


class TripletHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(3, NEG_FEAT))
        self.fc = nn.Linear(NEG_FEAT * PRIOR_CH, FC_DIM)
        self.cls_block = nn.Sequential(nn.Linear(FC_DIM, FC_DIM), nn.ReLU())
        self.reg_block = nn.Sequential(nn.Linear(FC_DIM, FC_DIM), nn.ReLU())
        self.cls_layer = nn.Linear(FC_DIM, 1)
        self.reg_layer = nn.Linear(FC_DIM, NUM_OFFSETS + 2)
        self.aux_reg_layer = nn.Linear(FC_DIM, 2 * NUM_GROUPS)       # only for strict load
        self.o2o_cls_head = O2OClsHead()

    def pool(self, feat_list, pts):
        g = (pts * 2.0 - 1.0).float()
        w = torch.softmax(self.layer_logits, 0)
        out = 0
        for i, f in enumerate(feat_list):
            s = F.grid_sample(f.float(), g, mode='bilinear', padding_mode='zeros', align_corners=True)
            out = out + (s * w[i]).permute(0, 2, 3, 1)
        return self.fc(out.reshape(pts.shape[0], pts.shape[1], -1))

    def forward(self, feat_list, pts, ids, emb):
        f = self.pool(feat_list, pts)
        cls = self.cls_layer(self.cls_block(f)).squeeze(-1).sigmoid()
        reg = self.reg_layer(self.reg_block(f))
        o2o = self.o2o_cls_head(f, cls, ids, emb)
        return cls, reg, o2o


class GlobalPolarHead(nn.Module):
    def __init__(self):
        super().__init__()
        n = NUM_OFFSETS - 1
        si = torch.linspace(0, 1, steps=NEG_FEAT, dtype=torch.float32)
        self.register_buffer('sample_index', torch.flip(n - (si * n).long(), dims=[-1]))
        self.rcnn_head = TripletHead()
        self.register_buffer('car_y', CENTER_H - torch.linspace(0, 1 - 1e-5, steps=NUM_OFFSETS) * IMG_H, persistent=False)

    def forward(self, feats, emb, ids, thr):
        feat_list = list(feats)[::-1]
        ang = emb[..., 0] * math.pi
        rho = emb[..., 1] * IMG_W
        B, N = ang.shape
        cy = self.car_y.view(1, 1, -1)
        cx = -torch.tan(ang).unsqueeze(-1) * cy + (rho / torch.cos(ang)).unsqueeze(-1)       # (B,N,72)
        samples_car = torch.stack((cx, cy.expand_as(cx)), -1)
        img_s = car2img(samples_car)
        grid = img_s[..., self.sample_index, :].clone()
        grid[..., 0] /= IMG_W; grid[..., 1] /= IMG_H
        base = torch.flip(samples_car, dims=[-2])
        cls, reg, o2o = self.rcnn_head(feat_list, grid, ids, emb)
        endp, xs_off = reg[..., 0:2], reg[..., 2:]
        base_x = base[..., 0] + xs_off * IMG_W
        lanes_car = torch.stack((base_x, base[..., 1]), -1)
        pts = car2img(lanes_car)                                                              # (B,N,72,2), bottom -> top
        keep = o2o[0] >= thr
        return (pts[0][keep].cpu().numpy(), endp[0][keep].cpu().numpy(), o2o[0][keep].cpu().numpy())


class PolarRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = load_dla()
        self.neck = FPN()
        self.rpn_head = LocalPolarHead()
        self.roi_head = GlobalPolarHead()

    @torch.no_grad()
    def forward(self, img, thr=CONF_NMSFREE, fp16=False, return_scores=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            feats = self.neck(self.backbone(img)[1:])
        feats = [f.float() for f in feats]
        emb, ids = self.rpn_head(feats)
        pts, endp, conf = self.roi_head(feats, emb, ids, thr)
        return decode(pts, endp, conf if return_scores else None)


def decode(pts, endp, conf=None):
    """Replicates GlobalPolarHead.get_lane_point; returns a list of (K,2) arrays in coordinates normalized to
    the 800x320 crop -> (x/800, y as a fraction of the original 1440-row image)."""
    ns = NUM_OFFSETS - 1
    lanes, scores = [], []
    for k, (p, e) in enumerate(zip(pts, endp)):
        xs, ys = p[:, 0].copy(), p[:, 1].copy()
        start = min(max(0, int(round(float(e[0]) * ns))), ns)
        end = min(int(round(float(e[1]) * ns)) - 1, NUM_OFFSETS - 1)
        # repo: the mask tests x in [0,1] (pixels) -> in practice invalidates all points before 'start'
        if start > 0:
            valid = ((xs[:start] >= 0.) & (xs[:start] <= 1.))[::-1].cumprod()[::-1].astype(bool)
            xs[:start][~valid] = -2
        xs[end + 1:] = -2
        m = xs >= 0
        xs, ys = xs[m][::-1], ys[m][::-1]
        if len(xs) <= 1:
            continue
        ys = (ys * ((ORI_H - CUT_H) / IMG_H) + CUT_H) / ORI_H
        lanes.append(np.stack((xs / IMG_W, ys), 1))
        if conf is not None:
            scores.append(float(conf[k]))
    return lanes if conf is None else (lanes, scores)


def load_model(path, device):
    net = PolarRCNN()
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    r = net.load_state_dict(ck.get('state_dict', ck), strict=True)
    print(f'state_dict (strict=True): missing={list(r.missing_keys)} unexpected={list(r.unexpected_keys)}')
    return net.to(device).eval()


class PolarLaneDetector:
    """Lane detector. __call__(frame_bgr) -> (lanes, scores): lanes = list of Nx2 float32 arrays
    (x, y) in frame pixels (top -> bottom), scores = confidence of each lane."""

    def __init__(self, weights=None, device=None, thr=CONF_NMSFREE, fp16=False, verbose=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thr, self.fp16 = thr, fp16 and self.device.type == 'cuda'
        weights = weights or os.path.join(HERE, '..', '..', 'models', 'polarrcnn_curvelanes_dla34.pth')
        if verbose:
            print(f'[lanes] loading {weights}')
        self.net = load_model(weights, self.device)

    @torch.no_grad()
    def __call__(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        y_off = int(round(H * CUT_FRAC))
        rgb = cv2.cvtColor(frame_bgr[y_off:], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        lanes, scores = self.net(x, self.thr, self.fp16, return_scores=True)
        out = [np.stack((p[:, 0] * W, p[:, 1] * H), 1).astype(np.float32) for p in lanes]
        return out, scores

