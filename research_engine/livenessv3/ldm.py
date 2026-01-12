#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passive Liveness Detection (3D) — MediaPipe + OpenCV + InsightFace (3D) + YOLO + CBAM Attention

Highlights
----------
- Uses InsightFace FaceAnalysis with **3D landmarks** as main passive liveness cues.
- Optional YOLO (Ultralytics) for robust face detection; fallback to MediaPipe/haar.
- MediaPipe FaceMesh optional (for better crop/tracking); primary geometry from InsightFace.
- Lightweight CBAM-based CNN branch for appearance cues (optional; load your checkpoint).
- Passive features:
  * 3D z-variance (flatness cue for print/monitor attacks)
  * Surface normal variance / consistency
  * 2D reprojection error (PnP consistency with 3D points)
  * FFT high-frequency ratio (screen moiré/replay cue)
  * Laplacian variance (blur)
  * YCrCb chroma stability
  * LBP histogram uniformity (skin micro-texture)
- Late-fusion scoring with tunable weights.

Usage
-----
1) Install:
   pip install -r requirements_3d.txt

2) Webcam, YOLO face model, and CBAM checkpoint:
   python liveness_passive_3d.py --source 0 --yolo-weights /path/to/yolov8n-face.pt --ckpt /path/to/liveness_cbam.pth

3) No YOLO (MediaPipe/haar only):
   python liveness_passive_3d.py --source 0 --no-yolo

Notes
-----
- Some InsightFace builds don't accept 'providers' arg on .prepare(). This script handles both via try/except.
- On Apple Silicon (M1/M2), ONNX Runtime uses CPU/CoreML providers; CUDA not available.
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# Optional imports guarded so the script still works in reduced mode.
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_AVAILABLE = False

MP_AVAILABLE = True
try:
    import mediapipe as mp
except Exception:
    MP_AVAILABLE = False

INSIGHT_AVAILABLE = True
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    INSIGHT_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    TORCH_AVAILABLE = False


# -------------------------
# Utils
# -------------------------

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def safe_div(a, b, eps=1e-8): return a / (b + eps)

def crop_with_margin(img, box, margin=0.15):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(margin * bw), int(margin * bh)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def laplacian_var(gray): return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def fft_high_ratio(gray, cutoff=0.25):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape; cy, cx = h//2, w//2
    ry, rx = int(cutoff*cy), int(cutoff*cx)
    low = mag[cy-ry:cy+ry, cx-rx:cx+rx].sum()
    total = mag.sum()
    high = total - low
    return float(safe_div(high, total))

def lbp_hist(gray, P=8, R=1):
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(R, h-R):
        for j in range(R, w-R):
            c = gray[i, j]; code = 0
            for p in range(P):
                theta = 2.0*math.pi*p/P
                y = i + int(round(R*math.sin(theta)))
                x = j + int(round(R*math.cos(theta)))
                code |= (gray[y, x] >= c) << p
            lbp[i, j] = code
    hist, _ = np.histogram(lbp, bins=256, range=(0,256), density=True)
    return hist.astype(np.float32)

def ycrcb_consistency(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    cr_std = float(np.std(Cr)); cb_std = float(np.std(Cb))
    scale = 1.0/128.0
    return cr_std*scale, cb_std*scale

def resize_pad(img, size=224):
    h, w = img.shape[:2]; s = size
    scale = s / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    y0 = (s - nh)//2; x0 = (s - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# -------------------------
# CBAM attention net (optional)
# -------------------------

if TORCH_AVAILABLE:
    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=8):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

    class SpatialAttention(nn.Module):
        def __init__(self, k=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            mx, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg, mx], dim=1)
            return self.sigmoid(self.conv(x))

    class CBAM(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ca = ChannelAttention(c)
            self.sa = SpatialAttention(7)

        def forward(self, x):
            x = self.ca(x) * x
            x = self.sa(x) * x
            return x

    class LivenessCBAMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                CBAM(32),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                CBAM(64),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                CBAM(128),
                nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                CBAM(256),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return torch.sigmoid(self.head(self.features(x))).squeeze(1)


# -------------------------
# Detection stacks
# -------------------------

class FaceDetector:
    def __init__(self, use_yolo=True, yolo_weights=None, conf_thres=0.25):
        self.use_yolo = use_yolo and YOLO_AVAILABLE and (yolo_weights is not None)
        self.conf_thres = conf_thres
        self.yolo = YOLO(yolo_weights) if self.use_yolo else None

        if MP_AVAILABLE:
            self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
            self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                           refine_landmarks=True, min_detection_confidence=0.4,
                                                           min_tracking_confidence=0.4)
        else:
            self.mp_face = None
            self.mp_mesh = None

    def detect(self, frame):
        h, w = frame.shape[:2]

        # YOLO
        if self.yolo is not None:
            res = self.yolo.predict(source=frame, verbose=False, imgsz=640)
            boxes = []
            for r in res:
                if r.boxes is None: continue
                for b in r.boxes:
                    conf = float(b.conf.cpu().numpy()[0])
                    if conf < self.conf_thres: continue
                    xyxy = b.xyxy.cpu().numpy()[0].astype(int)
                    boxes.append(xyxy.tolist())
            if boxes: return boxes

        # MediaPipe fallback
        if self.mp_face is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self.mp_face.process(rgb)
            boxes = []
            if out.detections:
                for det in out.detections:
                    rb = det.location_data.relative_bounding_box
                    x1 = int(rb.xmin * w); y1 = int(rb.ymin * h)
                    x2 = int((rb.xmin + rb.width) * w); y2 = int((rb.ymin + rb.height) * h)
                    boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
            if boxes: return boxes

        # Haar fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rects = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return [[x, y, x+w_, y+h_] for (x, y, w_, h_) in rects]

    def mesh(self, frame):
        if self.mp_mesh is None: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self.mp_mesh.process(rgb)
        if out.multi_face_landmarks:
            return out.multi_face_landmarks[0]
        return None


# -------------------------
# InsightFace 3D adapter
# -------------------------

class Insight3DAdapter:
    """
    Wrap InsightFace FaceAnalysis and expose 3D cues:
      - 3D landmarks (68) & projected 2D
      - Pose (r,p,y) in degrees
      - Z variance, surface normal variance
      - Reprojection error via solvePnP
    """
    def __init__(self, det_size=(640, 640), providers=None):
        self.ok = INSIGHT_AVAILABLE
        if not self.ok:
            self.app = None; return

        self.app = FaceAnalysis(name='buffalo_l')
        ctx = 0 if self._has_cuda() else -1
        try:
            self.app.prepare(ctx_id=ctx, det_size=det_size, providers=providers)
        except TypeError:
            self.app.prepare(ctx_id=ctx, det_size=det_size)

        # Approx intrinsics for 112x112 crops; tune if you have calibration.
        self.fx = 800.0; self.fy = 800.0; self.cx = 112.0; self.cy = 112.0
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0,      0,     1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1), dtype=np.float32)

    def _has_cuda(self):
        try:
            import torch; return torch.cuda.is_available()
        except Exception:
            return False

    def get(self, bgr):
        if not self.ok: return None
        faces = self.app.get(bgr)
        return None if len(faces) == 0 else faces[0]

    def cues(self, bgr):
        """
        Returns dict with:
          det_score, pose_deg (r,p,y),
          z_var, normal_var, reproj_err,
          emb_norm
        """
        f = self.get(bgr)
        if f is None:
            return dict(det=0.2, pose=(0,0,0), z_var=0.0, n_var=0.0, reproj=1.0, emb=0.4)

        det_score = float(getattr(f, 'det_score', 0.6))
        pose = getattr(f, 'pose', (0.0, 0.0, 0.0))
        r, p, y = [float(x) for x in pose]

        # 3D landmarks
        lmk3d = getattr(f, 'landmark_3d_68', None)
        lmk2d = getattr(f, 'landmark_2d_106', None)
        emb = getattr(f, 'embedding', None)
        emb_norm = 0.5 if emb is None else float(min(1.0, np.linalg.norm(emb) / 30.0))

        z_var = 0.0; normal_var = 0.0; reproj = 0.8
        if lmk3d is not None:
            P3 = np.array(lmk3d, dtype=np.float32)  # (68,3)
            z = P3[:,2]
            z_var = float(np.clip(np.var(z) / 2000.0, 0.0, 1.0))

            normals = []
            for i in range(2, P3.shape[0] - 2, 3):
                v1 = P3[i] - P3[i-1]; v2 = P3[i+1] - P3[i]
                n = np.cross(v1, v2); n = n / (np.linalg.norm(n) + 1e-6)
                normals.append(n)
            if normals:
                normals = np.array(normals)
                normal_var = float(np.clip(np.var(normals, axis=0).mean(), 0.0, 1.0))

            if lmk2d is not None and len(lmk2d) >= 68:
                P2 = np.array(lmk2d[:68], dtype=np.float32)
                ok, rvec, tvec = cv2.solvePnP(P3, P2, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    proj, _ = cv2.projectPoints(P3, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    proj = proj.reshape(-1,2)
                    err = np.linalg.norm(P2 - proj, axis=1).mean()
                    reproj = float(np.clip(err / 10.0, 0.0, 1.0))

        return dict(det=det_score, pose=(r,p,y), z_var=z_var, n_var=normal_var, reproj=reproj, emb=emb_norm)


# -------------------------
# Fusion logic
# -------------------------

class Fusion3D:
    """
    Combine:
      - CNN prob (optional)
      - 3D cues: z_var (↑ live), normal_var (↑ live), reproj (↓ is better → we invert), pose magnitude
      - Classical cues: FFT ratio, Laplacian, LBP uniformity, YCrCb stability
      - Insight det/emb
    """
    def __init__(self, use_cnn=True):
        self.use_cnn = use_cnn
        self.w = {
            'bias': -0.8,
            'cnn': 2.0,
            'zv': 1.2,
            'nv': 0.8,
            'reproj_inv': 1.0,
            'pose': 0.2,
            'fft': 0.6,
            'lap': 0.6,
            'lbp': 0.5,
            'cr': 0.25,
            'cb': 0.25,
            'det': 0.4,
            'emb': 0.3,
        }

    def score(self, feats):
        x = self.w['bias']
        if self.use_cnn: x += self.w['cnn'] * feats['cnn']
        x += self.w['zv'] * feats['z_var']
        x += self.w['nv'] * feats['n_var']
        x += self.w['reproj_inv'] * (1.0 - feats['reproj'])
        x += self.w['pose'] * feats['pose']
        x += self.w['fft'] * feats['fft']
        x += self.w['lap'] * feats['lap']
        x += self.w['lbp'] * feats['lbp']
        x += self.w['cr'] * feats['cr']
        x += self.w['cb'] * feats['cb']
        x += self.w['det'] * feats['det']
        x += self.w['emb'] * feats['emb']
        return float(sigmoid(x))


# -------------------------
# Main
# -------------------------

def run(args):
    # Prepare stacks
    detector = FaceDetector(use_yolo=(not args.no_yolo), yolo_weights=args.yolo_weights, conf_thres=args.conf_thres)
    insight3d = Insight3DAdapter()

    device = 'cpu'; net = None
    if TORCH_AVAILABLE and args.ckpt and Path(args.ckpt).exists():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = LivenessCBAMNet().to(device)
        state = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()
        print(f'[INFO] Loaded CBAM checkpoint from {args.ckpt} on {device}')
    else:
        if TORCH_AVAILABLE and args.ckpt: print('[WARN] CKPT not found; CNN branch disabled.')
        elif not TORCH_AVAILABLE: print('[WARN] Torch not available; CNN branch disabled.')

    fuse = Fusion3D(use_cnn=(net is not None))

    # Source
    src = 0 if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(int(src)) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open source: {args.source}'); sys.exit(1)

    win = 'Passive Liveness 3D (q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_t0 = time.time(); frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames += 1

        boxes = detector.detect(frame)
        if len(boxes) == 0:
            cv2.putText(frame, 'No face', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            continue

        # choose largest face
        areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in boxes ]
        b = boxes[int(np.argmax(areas))]
        crop, (x1,y1,x2,y2) = crop_with_margin(frame, b, margin=0.12)
        if crop.size == 0:
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            continue

        face224 = resize_pad(crop, 224)
        gray = cv2.cvtColor(face224, cv2.COLOR_BGR2GRAY)

        # --- InsightFace 3D cues
        cues = insight3d.cues(face224)
        det_score = cues['det']
        r,p,yaw = cues['pose']
        pose_mag = float(min(1.0, (abs(r)+abs(p)+abs(yaw))/90.0))
        z_var = float(np.clip(cues['z_var'], 0.0, 1.0))
        n_var = float(np.clip(cues['n_var'], 0.0, 1.0))
        reproj = float(np.clip(cues['reproj'], 0.0, 1.0))
        emb_norm = cues['emb']

        # --- Classical passive cues
        fft_ratio = float(np.clip(fft_high_ratio(gray), 0.0, 1.0))
        lap = float(np.clip(laplacian_var(gray) / 300.0, 0.0, 1.0))
        lbp_u = float(np.clip(1.0 - np.std(lbp_hist(gray)), 0.0, 1.0))
        cr_std, cb_std = ycrcb_consistency(face224)
        cr_std = float(np.clip(cr_std, 0.0, 1.0))
        cb_std = float(np.clip(cb_std, 0.0, 1.0))

        # --- CNN
        cnn_prob = 0.0
        if net is not None:
            with torch.no_grad():
                t = torch.from_numpy(face224[:, :, ::-1].transpose(2,0,1)).float().unsqueeze(0) / 255.0
                t = t.to(device)
                cnn_prob = float(net(t).cpu().numpy()[0])

        feats = {
            'cnn': cnn_prob,
            'z_var': z_var,
            'n_var': n_var,
            'reproj': reproj,
            'pose': pose_mag,
            'fft': fft_ratio,
            'lap': lap,
            'lbp': lbp_u,
            'cr': cr_std,
            'cb': cb_std,
            'det': det_score,
            'emb': emb_norm,
        }

        prob_live = fuse.score(feats)
        label = 'LIVE' if prob_live >= args.threshold else 'SPOOF'
        color = (40, 200, 40) if label == 'LIVE' else (30, 30, 230)

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt1 = f'{label}  p={prob_live:.2f}  CNN={feats["cnn"]:.2f} FFT={feats["fft"]:.2f} LBP={feats["lbp"]:.2f} LAP={feats["lap"]:.2f}'
        txt2 = f'3D zv={feats["z_var"]:.2f} nv={feats["n_var"]:.2f} reproj={feats["reproj"]:.2f} pose={feats["pose"]:.2f} det={feats["det"]:.2f}'
        cv2.putText(frame, txt1, (x1, max(20, y1-26)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        cv2.putText(frame, txt2, (x1, max(44, y1-6)),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        # FPS overlay every 20 frames
        if frames % 20 == 0:
            now = time.time(); fps = 20.0 / (now - fps_t0); fps_t0 = now
            cv2.putText(frame, f'FPS: {fps:.1f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break

    cap.release(); cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description='Passive Liveness 3D with InsightFace + MediaPipe + YOLO + CBAM')
    ap.add_argument('--source', type=str, default='0', help='Camera index or video path')
    ap.add_argument('--yolo-weights', type=str, default=None, help='Path to YOLOv8 face weights (e.g., yolov8n-face.pt)')
    ap.add_argument('--no-yolo', action='store_true', help='Disable YOLO detector')
    ap.add_argument('--conf-thres', type=float, default=0.25, help='YOLO conf threshold')
    ap.add_argument('--ckpt', type=str, default=None, help='CBAM liveness checkpoint (.pth)')
    ap.add_argument('--threshold', type=float, default=0.60, help='LIVE threshold')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
