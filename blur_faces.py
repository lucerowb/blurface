#!/usr/bin/env python3
"""
Blurface - keep chosen people visible, blur every other face for the WHOLE video.

How it works
  1. Analyze: every frame is scanned with YuNet (face detector). Each face gets a
     SFace embedding (a numeric "fingerprint"). Detections are linked into tracks
     over time and tracks are grouped into people by fingerprint similarity.
  2. You tick the people to KEEP visible. Everyone else (and anything that could
     not be identified) is blurred.
  3. Export: blur boxes are filled across detector gaps and extended before/after
     each appearance so a missed detection does not leak a face. Audio from the
     original is re-attached.

Install
  pip install -r requirements.txt
  (Tkinter is also required; on macOS/Homebrew: brew install python-tk@3.12)

Run
  python blur_faces.py

The two small ONNX models (~38 MB total) are downloaded once to ~/.cache/blurface.
"""

import base64
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from collections import defaultdict

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # lets the core logic be imported/tested without Tk
    tk = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "blurface")
MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}

EMBED_MIN_PX = 40      # faces smaller than this are not fingerprinted (-> blurred)
EMBED_MIN_SCORE = 0.7  # low-confidence detections are not fingerprinted
LINK_MIN_IOU = 0.15    # min box overlap to link detections into a track
LINK_MIN_SIM = 0.15    # below this fingerprint similarity, never link (identity swap guard)
VERIFY_MIN_SIM = 0.20  # a "kept" detection that looks this unlike the kept person is blurred

CRF_MIN, CRF_MAX, CRF_DEFAULT = 18, 36, 28
QUALITY_PRESETS = (("Small", 34), ("Compact", 28), ("High", 23), ("Max", 18))
RESOLUTION_PRESETS = (("Original", 0), ("1080p", 1920), ("720p", 1280), ("480p", 854))

BG = "#f4f1ea"
CARD = "#fffcf8"
INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e6e0d6"
ACCENT = "#0c6b58"
ACCENT_PRESS = "#095445"
SOFT = "#efeae2"
WARN = "#9a3412"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _download(url, dst):
    tmp = dst + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        curl = shutil.which("curl")
        if not curl:
            raise
        subprocess.run([curl, "-L", "--fail", "-s", "-o", tmp, url], check=True)
    os.replace(tmp, dst)


def ensure_models(status=print):
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    for key, (name, url) in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(path) or os.path.getsize(path) < 100_000:
            status(f"Downloading model {name} ...")
            try:
                _download(url, path)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download {name}.\nDownload it manually with:\n"
                    f"  mkdir -p '{MODEL_DIR}' && curl -L -o '{path}' '{url}'\n({e})")
            if os.path.getsize(path) < 100_000:
                os.remove(path)
                raise RuntimeError(f"Downloaded {name} looks invalid; try the manual curl download.")
        paths[key] = path
    return paths


class FaceEngine:
    """YuNet detector + SFace recognizer (both ship with plain opencv-python)."""

    def __init__(self, score_threshold=0.5, status=print):
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("Your OpenCV is too old. Run: pip install -U opencv-python")
        paths = ensure_models(status)
        self.det = cv2.FaceDetectorYN.create(paths["yunet"], "", (320, 320),
                                             float(score_threshold), 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(paths["sface"], "")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        return [] if faces is None else list(faces)

    def embed(self, frame, row):
        try:
            aligned = self.rec.alignCrop(frame, row)
            f = self.rec.feature(aligned).flatten().astype(np.float32)
        except cv2.error:
            return None
        n = float(np.linalg.norm(f))
        return f / n if n > 0 else None


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def face_is_plausible(row, box):
    """Reject hands, phones, and other boxes that only loosely look like a face.

    YuNet always emits five landmarks. A real face has the eyes above the nose
    and the nose above the mouth, inside a roughly face-shaped box.
    """
    if row is None or len(row) < 15 or box is None:
        return False
    x, y, w, h = box
    if w < 20 or h < 20:
        return False
    aspect = w / h
    if aspect < 0.6 or aspect > 1.45:
        return False
    pts = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
    margin_x, margin_y = 0.25 * w, 0.3 * h
    for px, py in pts:
        if px < x - margin_x or px > x + w + margin_x:
            return False
        if py < y - margin_y or py > y + h + margin_y:
            return False
    right_eye, left_eye, nose, right_mouth, left_mouth = pts
    eye_y = (right_eye[1] + left_eye[1]) * 0.5
    mouth_y = (right_mouth[1] + left_mouth[1]) * 0.5
    if not (eye_y + 0.02 * h < nose[1] < mouth_y - 0.02 * h):
        return False
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 0.15 * w or eye_dist > 0.85 * w:
        return False
    if abs(right_eye[1] - left_eye[1]) > 0.35 * h:
        return False
    mouth_dist = abs(right_mouth[0] - left_mouth[0])
    if mouth_dist < 0.1 * w:
        return False
    eye_left = min(right_eye[0], left_eye[0])
    eye_right = max(right_eye[0], left_eye[0])
    if nose[0] < eye_left - 0.15 * w or nose[0] > eye_right + 0.15 * w:
        return False
    span = mouth_y - eye_y
    if span < 0.18 * h or span > 0.72 * h:
        return False
    return True


def looks_like_skin(frame, box, min_fraction=0.16):
    """True when the middle of the box has face-like skin color.

    A phone, bag, or other object fails this. A very confident detection
    skips the check, so unusual lighting can still keep a real face.
    """
    H, W = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in box[:4]]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + max(w, 1)), min(H, y + max(h, 1))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    rh, rw = roi.shape[:2]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 183) & (cb >= 77) & (cb <= 135)
    mask = np.zeros((rh, rw), np.uint8)
    cv2.ellipse(
        mask,
        (rw // 2, rh // 2),
        (max(1, int(rw * 0.34)), max(1, int(rh * 0.40))),
        0, 0, 360, 1, thickness=-1,
    )
    area = int(mask.sum())
    if area < 8:
        return False
    return float((skin & (mask > 0)).sum()) / area >= min_fraction


def body_false_positive(box, score, faces):
    """A box on the hands or an object held under a clearer face."""
    ox, oy, ow, oh = box
    other_cy = oy + oh * 0.5
    for fx, fy, fw, fh, face_score in faces:
        if face_score < 0.82 or score >= face_score:
            continue
        overlap = min(fx + fw, ox + ow) - max(fx, ox)
        if overlap < 0.4 * min(fw, ow):
            continue
        if oy < fy + fh * 0.55:
            continue
        if other_cy - (fy + fh * 0.5) > 2.2 * fh:
            continue
        return True
    return False


def track_should_blur(track, fps):
    """Keep a track only when it is confident or clearly persistent."""
    if not track.dets:
        return False
    scores = [d.score for d in track.dets]
    mean = sum(scores) / len(scores)
    peak = max(scores)
    seconds = len(scores) / max(float(fps or 1), 1.0)
    if peak >= 0.9:
        return True
    if mean >= 0.8 and seconds >= 0.15:
        return True
    if mean >= 0.74 and seconds >= 0.4:
        return True
    return False


def clip_box(row, W, H):
    x, y, w, h = [float(v) for v in row[:4]]
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(W), x + w), min(float(H), y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def lerp_box(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def square_crop(frame, box, scale=1.6, out=96):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    s = max(8, int(round(max(w, h) * scale)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = x0 + s, y0 + s
    H, W = frame.shape[:2]
    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if vx1 <= vx0 or vy1 <= vy0:
        return np.zeros((out, out, 3), np.uint8)
    crop = frame[vy0:vy1, vx0:vx1]
    crop = cv2.copyMakeBorder(crop, vy0 - y0, y1 - vy1, vx0 - x0, x1 - vx1,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def timecode(seconds):
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def format_size(num_bytes):
    if num_bytes is None:
        return "—"
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def format_fps(fps):
    if abs(fps - round(fps)) < 0.05:
        return f"{fps:.0f} fps"
    return f"{fps:.2f} fps"


def quality_name(crf):
    crf = int(crf)
    if crf <= 20:
        return "Max"
    if crf <= 24:
        return "High"
    if crf <= 29:
        return "Compact"
    if crf <= 33:
        return "Small"
    return "Smallest"


def audio_bitrate(crf):
    if crf <= 20:
        return "192k"
    if crf <= 29:
        return "128k"
    return "96k"


def fit_dimensions(width, height, max_edge):
    """Even output size. max_edge 0 keeps the original long side."""
    width = max(2, int(width))
    height = max(2, int(height))
    long_side = max(width, height)
    if max_edge and long_side > max_edge:
        scale = max_edge / float(long_side)
        width = int(round(width * scale))
        height = int(round(height * scale))
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    return width, height


def estimate_output_bytes(width, height, n_frames, fps, crf, max_edge):
    """Rough H.264 size. Anchored near 4.5 Mbps for 1080p30 at CRF 23."""
    w, h = fit_dimensions(width, height, max_edge)
    bpp = 0.072 * (2 ** ((23 - float(crf)) / 6.0))
    if fps and fps > 40:
        bpp *= 0.85
    frames = max(1, int(n_frames))
    video = w * h * frames * bpp / 8.0
    duration = frames / max(float(fps or 30.0), 1.0)
    rate = int(audio_bitrate(crf).rstrip("k")) * 1000
    return max(1, int(video + (rate / 8.0) * duration + 2048))


def comparison_sentence(out_bytes, in_bytes, estimated=True):
    if not in_bytes or not out_bytes:
        return "Choose a video to estimate the output size."
    pct = 100.0 * out_bytes / float(in_bytes)
    lead = "About " if estimated else ""
    if pct >= 105:
        return (f"{lead}{pct:.0f}% of the input. Choose Small, or a lower "
                "resolution, to bring the file down.")
    if pct >= 98:
        return f"{lead}the same size as the input."
    return f"{lead}{pct:.0f}% of the input · {100 - pct:.0f}% smaller."


def wheel_steps(delta=0, button=0):
    """Convert a mouse-wheel event into canvas scroll units.

    macOS reports a small delta (often ±1). Windows and Linux report ±120 per notch.
    """
    if button == 4:
        return -1
    if button == 5:
        return 1
    delta = int(delta or 0)
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        steps = int(-delta / 120)
    else:
        steps = -delta
    if steps == 0:
        steps = -1 if delta > 0 else 1
    return steps


def shorten_path(path, limit=52):
    if not path or len(path) <= limit:
        return path or ""
    keep = limit - 3
    head = keep // 2
    tail = keep - head
    return path[:head] + "..." + path[-tail:]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
class Det:
    __slots__ = ("f", "box", "score", "emb")

    def __init__(self, f, box, score, emb):
        self.f, self.box, self.score, self.emb = f, box, score, emb


class Track:
    def __init__(self, tid):
        self.id = tid
        self.dets = []
        self.last_frame = -1
        self.last_box = None
        self.last_emb = None
        self.n_emb = 0
        self.best_q = -1.0
        self.best_crop = None
        self.mean_emb = None
        self.weight = 0.0
        self.person = None

    def add(self, f, box, score, emb, frame):
        self.dets.append(Det(f, box, score, emb))
        self.last_frame, self.last_box = f, box
        if emb is not None:
            self.last_emb = emb
            self.n_emb += 1
        q = score * min(box[2], box[3])
        if score >= EMBED_MIN_SCORE and q > self.best_q * 1.15:
            self.best_q = q
            self.best_crop = square_crop(frame, box)

    def finalize(self):
        acc, wsum = None, 0.0
        for d in self.dets:
            if d.emb is not None:
                acc = d.emb * d.score if acc is None else acc + d.emb * d.score
                wsum += d.score
        if acc is not None and np.linalg.norm(acc) > 0:
            self.mean_emb, self.weight = _unit(acc), wsum


class Person:
    def __init__(self, centroid):
        self.id = 0
        self.centroid = centroid
        self.tracks = []

    @property
    def frames(self):
        return sum(len(t.dets) for t in self.tracks)

    @property
    def first_frame(self):
        return min(t.dets[0].f for t in self.tracks)

    def thumbs(self, n=3):
        ts = sorted((t for t in self.tracks if t.best_crop is not None),
                    key=lambda t: -t.best_q)
        return [t.best_crop for t in ts[:n]]


class Analysis:
    def __init__(self, path, tracks, fps, size, n_frames):
        self.path, self.tracks, self.fps = path, tracks, fps
        self.size, self.n_frames = size, n_frames


def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps == fps and 1 <= fps <= 240 else 30.0


# --------------------------------------------------------------------------
# Pass 1: detect every frame, track, fingerprint
# --------------------------------------------------------------------------
def analyze_video(path, engine, progress, cancel, gap_seconds=1.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = get_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gap_frames = max(2, int(round(fps * gap_seconds)))

    tracks, active = [], []
    idx, size = 0, None
    t0 = time.time()
    while not cancel.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        size = (W, H)

        rows = engine.detect(frame)
        boxes = [clip_box(r, W, H) for r in rows]
        faces = []
        for row, box in zip(rows, boxes):
            if box is None or len(row) < 15 or not face_is_plausible(row, box):
                continue
            if float(row[14]) >= 0.82:
                faces.append((*box, float(row[14])))
        active = [t for t in active if idx - t.last_frame <= gap_frames]

        # greedy IoU linking to recently-seen tracks
        cands = []
        for di, b in enumerate(boxes):
            if b is None:
                continue
            for ti, t in enumerate(active):
                v = iou(b, t.last_box)
                if v >= LINK_MIN_IOU:
                    cands.append((v, di, ti))
        cands.sort(reverse=True)
        used_d, used_t, assigned = set(), set(), {}
        for _, di, ti in cands:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            assigned[di] = active[ti]

        for di, row in enumerate(rows):
            box = boxes[di]
            if box is None:
                continue
            if len(row) < 15 or not face_is_plausible(row, box):
                continue
            score = float(row[14])
            if body_false_positive(box, score, faces):
                continue
            if score < 0.9 and not looks_like_skin(frame, box):
                continue
            tr = assigned.get(di)
            emb = None
            eligible = score >= EMBED_MIN_SCORE and min(box[2], box[3]) >= EMBED_MIN_PX
            if eligible and (tr is None or tr.n_emb < 10 or idx % 3 == 0):
                emb = engine.embed(frame, row)
            # identity-swap guard: same place but clearly a different face -> new track
            if (tr is not None and emb is not None and tr.last_emb is not None
                    and float(emb @ tr.last_emb) < LINK_MIN_SIM):
                tr = None
            if tr is None:
                tr = Track(len(tracks))
                tracks.append(tr)
            tr.add(idx, box, score, emb, frame)
            if tr not in active:
                active.append(tr)

        idx += 1
        if idx % 10 == 0:
            frac = (idx / total) if total else 0.0
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx) if total else 0
            progress(min(frac, 1.0),
                     f"Analyzing frame {idx}/{total or '?'}"
                     + (f" - about {timecode(eta)} left" if total else ""))
    cap.release()

    for t in tracks:
        t.finalize()
    return Analysis(path, tracks, fps, size, idx)


# --------------------------------------------------------------------------
# Group tracks into people
# --------------------------------------------------------------------------
def cluster_tracks(tracks, thr):
    for t in tracks:
        t.person = None
    items = sorted([t for t in tracks if t.mean_emb is not None], key=lambda t: -t.weight)
    cents, sums, groups = [], [], []
    for t in items:
        j = -1
        if cents:
            sims = np.stack(cents) @ t.mean_emb
            k = int(np.argmax(sims))
            if sims[k] >= thr:
                j = k
        if j < 0:
            cents.append(t.mean_emb.copy())
            sums.append(t.mean_emb * t.weight)
            groups.append([t])
        else:
            groups[j].append(t)
            sums[j] = sums[j] + t.mean_emb * t.weight
            cents[j] = _unit(sums[j])
    for _ in range(2):  # refinement: reassign everyone to the nearest final centroid
        if not cents:
            break
        C = np.stack(cents)
        regroup = [[] for _ in cents]
        for t in items:
            regroup[int(np.argmax(C @ t.mean_emb))].append(t)
        groups = [g for g in regroup if g]
        cents = [_unit(sum(t.mean_emb * t.weight for t in g)) for g in groups]

    persons = []
    for g, c in zip(groups, cents):
        p = Person(c)
        p.tracks = g
        persons.append(p)
    persons.sort(key=lambda p: -p.frames)
    for i, p in enumerate(persons, 1):
        p.id = i
        for t in p.tracks:
            t.person = p
    return persons


# --------------------------------------------------------------------------
# Decide what to blur on every frame (the "no gaps" logic)
# --------------------------------------------------------------------------
def add_track_boxes(per_frame, dets, gap_frames, extend):
    first, last = dets[0], dets[-1]
    for f in range(max(0, first.f - extend), first.f):          # lead-in
        per_frame[f].append(first.box)
    for i, d in enumerate(dets):
        per_frame[d.f].append(d.box)
        if i + 1 < len(dets):
            n = dets[i + 1]
            gap = n.f - d.f
            if gap <= 1:
                continue
            if gap <= gap_frames:                                # fill missed frames
                for f in range(d.f + 1, n.f):
                    per_frame[f].append(lerp_box(d.box, n.box, (f - d.f) / gap))
            else:
                for f in range(d.f + 1, min(d.f + 1 + extend, n.f)):
                    per_frame[f].append(d.box)
                for f in range(max(d.f + 1, n.f - extend), n.f):
                    per_frame[f].append(n.box)
    for f in range(last.f + 1, last.f + 1 + extend):             # lead-out
        per_frame[f].append(last.box)


def compute_blur_boxes(analysis, keep_ids):
    fps = analysis.fps
    gap_frames = max(2, int(round(fps * 1.0)))
    extend = max(2, int(round(fps * 0.5)))
    per_frame = defaultdict(list)
    for t in analysis.tracks:
        p = t.person
        if p is not None and p.id in keep_ids:
            # A kept person stays visible. Only a high-confidence face that
            # clearly is someone else, and is not an object under their face,
            # is still covered.
            by_frame = defaultdict(list)
            for d in t.dets:
                if d.score >= 0.82:
                    by_frame[d.f].append((d.box, d.score))
            for d in t.dets:
                if d.emb is None or d.score < 0.88:
                    continue
                if float(d.emb @ p.centroid) >= VERIFY_MIN_SIM:
                    continue
                others = [face for face in by_frame[d.f] if face[0] is not d.box]
                if body_false_positive(d.box, d.score, [(b[0], b[1], b[2], b[3], s) for b, s in others]):
                    continue
                per_frame[d.f].append(d.box)
            continue
        if not track_should_blur(t, fps):
            continue
        add_track_boxes(per_frame, t.dets, gap_frames, extend)
    return per_frame


def resolve_shape(shape, box):
    if shape and shape != "Auto":
        return shape
    w, h = float(box[2]), float(box[3])
    if h <= 1:
        return "Ellipse"
    ratio = w / h
    if 0.82 <= ratio <= 1.22:
        return "Circle"
    return "Ellipse"


def _shape_mask(height, width, kind, box, origin, pad):
    """1 inside the face shape, fading to 0 across the padding outside it."""
    x, y, w, h = [float(v) for v in box[:4]]
    ox, oy = origin
    feather = max(8.0, 0.2 * max(w, h))
    mask = np.zeros((height, width), np.float32)
    if kind == "Rectangle":
        x0 = int(round(x - w * pad * 0.35 - ox))
        y0 = int(round(y - h * pad * 0.35 - oy))
        x1 = int(round(x + w + w * pad * 0.35 - ox))
        y1 = int(round(y + h + h * pad * 0.35 - oy))
        cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0, thickness=-1)
    else:
        cx = x + w / 2.0 - ox
        cy = y + h / 2.0 - oy
        if kind == "Circle":
            radius = 0.5 * max(w, h) * (1.0 + pad * 0.25)
            axes = (max(1, int(round(radius))), max(1, int(round(radius))))
        else:
            axes = (max(1, int(round(w * (0.55 + pad * 0.15)))),
                    max(1, int(round(h * (0.62 + pad * 0.15)))))
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), axes, 0, 0, 360, 1.0, thickness=-1)
    core = (mask > 0.5).astype(np.uint8)
    outside = cv2.distanceTransform(((1 - core) * 255).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(1.0 - outside / feather, 0.0, 1.0)
    alpha[core > 0] = 1.0
    return alpha


def blur_box(frame, box, style, pad, shape="Auto"):
    H, W = frame.shape[:2]
    x, y, w, h = [float(v) for v in box[:4]]
    kind = resolve_shape(shape, box)
    feather = max(8.0, 0.2 * max(w, h))
    reach = max(w, h) * (0.65 + pad) + feather
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = max(0, int(cx - reach)), max(0, int(cy - reach))
    x1, y1 = min(W, int(cx + reach)), min(H, int(cy + reach))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    alpha = _shape_mask(rh, rw, kind, box, (x0, y0), pad)
    if float(alpha.max()) <= 0:
        return
    painted = roi.copy()
    if style == "Black box":
        painted[:] = 0
    elif style == "Pixelate":
        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
        painted[:] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        sigma = max(12.0, max(w, h) / 5.0)
        painted = cv2.GaussianBlur(roi, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    a = alpha[:, :, None]
    blended = painted.astype(np.float32) * a + roi.astype(np.float32) * (1.0 - a)
    roi[:] = np.clip(blended, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Pass 2: render + audio
# --------------------------------------------------------------------------
def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def mux_audio(tmp_video, src, dst, crf):
    """Encode H.264 at the chosen CRF and attach audio. Returns (note, error_or_None)."""
    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    crf = str(max(CRF_MIN, min(CRF_MAX, int(crf))))
    bitrate = audio_bitrate(int(crf))
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.move(tmp_video, dst)
        return ("Audio is missing from this export. Install imageio-ffmpeg and export again.", None)
    probe = subprocess.run([ffmpeg, "-i", src], capture_output=True, text=True).stderr
    has_audio = "Audio:" in probe
    attempts = [
        ["-c:v", "libx264", "-preset", "medium", "-crf", crf, "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", bitrate],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p",
         "-c:a", "copy"],
        ["-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-b:a", bitrate],
    ]
    last_err = ""
    for opts in attempts:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_video, "-i", src,
               "-map", "0:v:0", "-map", "1:a?", "-shortest", "-map_metadata", "-1",
               *opts, "-movflags", "+faststart", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
            if has_audio:
                return (f"Audio re-encoded at {bitrate}.", None)
            return ("Source had no audio track.", None)
        last_err = (r.stderr or "")[-400:]
    shutil.move(tmp_video, dst)
    return ("ffmpeg could not compress the file, so the large preview file was kept.", last_err)


def open_writer(directory, fps, size):
    attempts = (
        ("video.mp4", "mp4v"),
        ("video.avi", "MJPG"),
        ("video.avi", "XVID"),
    )
    for name, code in attempts:
        path = os.path.join(directory, name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), float(fps), size)
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("Could not open a video writer.")


def render_video(src, dst, per_frame, style, pad, fps, progress, cancel, total_hint,
                 crf=CRF_DEFAULT, max_edge=0, shape="Auto"):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    tmp_dir = tempfile.mkdtemp(prefix="blurface_")
    writer, tmp_video = None, None
    writer_size = None
    idx, blurred_frames = 0, 0
    try:
        while not cancel.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            orig_h, orig_w = frame.shape[:2]
            out_w, out_h = fit_dimensions(orig_w, orig_h, max_edge)
            if (orig_w, orig_h) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer_size = (frame.shape[1], frame.shape[0])
                writer, tmp_video = open_writer(tmp_dir, fps, writer_size)
            elif (frame.shape[1], frame.shape[0]) != writer_size:
                frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_AREA)
            sx = frame.shape[1] / float(orig_w)
            sy = frame.shape[0] / float(orig_h)
            boxes = per_frame.get(idx)
            if boxes:
                blurred_frames += 1
                for b in boxes:
                    if sx != 1.0 or sy != 1.0:
                        b = (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
                    blur_box(frame, b, style, pad, shape)
            writer.write(frame)
            idx += 1
            if idx % 15 == 0:
                progress(min(idx / total_hint, 1.0) if total_hint else 0.0,
                         f"Rendering frame {idx}/{total_hint or '?'}")
        if writer is not None:
            writer.release()
            writer = None
        if cancel.is_set():
            return None
        if idx == 0 or not tmp_video:
            raise RuntimeError("This video has no readable frames.")
        progress(1.0, "Compressing and adding audio ...")
        note, err = mux_audio(tmp_video, src, dst, crf)
        output_bytes = os.path.getsize(dst) if os.path.isfile(dst) else None
        return {"frames": idx, "blurred_frames": blurred_frames, "audio": note,
                "ffmpeg_error": err, "output_bytes": output_bytes}
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def probe_video(path):
    """Read size from the first decoded frame so phone rotation is included."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = get_fps(cap)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    try:
        nbytes = os.path.getsize(path)
    except OSError:
        nbytes = 0
    frames = n if n > 0 else 1
    return {
        "path": path,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "bytes": nbytes,
        "duration": frames / fps if fps else 0.0,
    }


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Blurface")
        root.geometry("1080x920")
        root.minsize(960, 780)
        root.configure(bg=BG)
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.analysis = None
        self.persons = []
        self.keep_vars = {}
        self.thumb_refs = []
        self.busy = False
        self.meta = None
        self.output_actual = None
        self.src = None
        self._bar_ratio = 0.0
        self._bar_warn = False
        self._after_id = None
        self.setting_scales = []
        self.quality_buttons = []
        self.resolution_buttons = []
        self.style_buttons = []
        self.shape_buttons = []
        self.crf = tk.IntVar(value=CRF_DEFAULT)
        self.max_edge = tk.IntVar(value=0)
        self.out_var = tk.StringVar()
        self.det_thr = tk.DoubleVar(value=0.72)
        self.shape = tk.StringVar(value="Auto")
        self.strict = tk.DoubleVar(value=0.40)
        self.margin = tk.DoubleVar(value=0.35)
        self.style = tk.StringVar(value="Strong blur")
        self.hide_brief = tk.BooleanVar(value=True)
        self._build()
        self._schedule()

    def _build(self):
        self._init_style()
        r = self.root
        footer = tk.Frame(r, bg=BG)
        footer.pack(side="bottom", fill="x")
        viewport = tk.Frame(r, bg=BG)
        viewport.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(viewport, bg=BG, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.page = tk.Frame(self.canvas, bg=BG)
        self.win = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self._page_width = None
        self.page.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._fit_page_width)
        body = self.page

        header = tk.Frame(body, bg=BG)
        header.pack(fill="x", padx=20, pady=(18, 8))
        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left", fill="x", expand=True)
        self._text(titles, "Blurface", size=26, bold=True).pack(anchor="w")
        self._text(titles, "Blur every face except the people you choose to keep.",
                   size=13, fg=MUTED, wrap=640).pack(anchor="w", pady=(2, 0))
        if sys.platform == "darwin":
            hint = "⌘O choose   ·   ⌘↩ export"
        else:
            hint = "Ctrl+O choose   ·   Ctrl+Enter export"
        self._text(titles, hint, size=11, fg=MUTED).pack(anchor="w", pady=(4, 0))
        pill = tk.Frame(header, bg="#e5f2ee")
        pill.pack(side="right", anchor="n")
        tk.Label(pill, text="On this computer", bg="#e5f2ee", fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(padx=10, pady=4)

        files = tk.Frame(body, bg=BG)
        files.pack(fill="x", padx=20, pady=(8, 0))
        files.columnconfigure(0, weight=1, uniform="cards")
        files.columnconfigure(1, weight=1, uniform="cards")
        files.rowconfigure(0, weight=1)
        in_outer, inn = self._card(files)
        out_outer, out = self._card(files)
        in_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        out_outer.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._text(inn, "File size is the video on disk, before any blur.",
                   size=11, fg=MUTED, wrap=420).pack(side="bottom", anchor="w", pady=(10, 0))
        in_head = tk.Frame(inn, bg=CARD)
        in_head.pack(fill="x")
        self._text(in_head, "INPUT", size=11, fg=MUTED, bold=True).pack(side="left")
        self._text(in_head, "On disk", size=11, fg=MUTED).pack(side="right")
        self.btn_open = self._button(inn, "Choose video", self.open_video, primary=True)
        self.btn_open.pack(anchor="w", pady=(10, 8))
        self.in_name = self._text(inn, "No video yet", size=15, bold=True)
        self.in_name.pack(anchor="w")
        self.in_path = self._text(inn, "Choose a video to see its size.", size=11, fg=MUTED, wrap=420)
        self.in_path.pack(anchor="w", pady=(2, 0))
        self.in_meta = self._text(inn, "Resolution, frame rate, and duration", size=12, fg=MUTED)
        self.in_meta.pack(anchor="w", pady=(8, 0))
        self.in_size = self._text(inn, "—", size=28, bold=True)
        self.in_size.pack(anchor="w", pady=(2, 0))

        out_head = tk.Frame(out, bg=CARD)
        out_head.pack(fill="x")
        self._text(out_head, "OUTPUT", size=11, fg=MUTED, bold=True).pack(side="left")
        self.out_kind = self._text(out_head, "Estimated", size=11, fg=MUTED)
        self.out_kind.pack(side="right")
        path_row = tk.Frame(out, bg=CARD)
        path_row.pack(fill="x", pady=(10, 0))
        self.out_entry = ttk.Entry(path_row, textvariable=self.out_var, font=("Helvetica", 12))
        self.out_entry.pack(side="left", fill="x", expand=True)
        self.btn_browse = self._button(path_row, "Save as", self.pick_output)
        self.btn_browse.pack(side="left", padx=(8, 0))

        comp = tk.Frame(out, bg=CARD)
        comp.pack(fill="x", pady=(12, 0))
        self._text(comp, "Compression", size=12, bold=True).pack(side="left")
        self.quality_caption = self._text(comp, "", size=12, fg=MUTED)
        self.quality_caption.pack(side="right")
        qrow, self.quality_buttons = self._choice_buttons(out, QUALITY_PRESETS, self.crf.set)
        qrow.pack(fill="x", pady=(6, 0))
        self.quality_scale = ttk.Scale(out, from_=CRF_MAX, to=CRF_MIN, variable=self.crf)
        self.quality_scale.pack(fill="x", pady=(4, 0))
        ends = tk.Frame(out, bg=CARD)
        ends.pack(fill="x")
        self._text(ends, "Smaller", size=11, fg=MUTED).pack(side="left")
        self._text(ends, "Sharper", size=11, fg=MUTED).pack(side="right")
        self._text(out, "Resolution", size=12, bold=True).pack(anchor="w", pady=(8, 0))
        rrow, self.resolution_buttons = self._choice_buttons(out, RESOLUTION_PRESETS, self.max_edge.set)
        rrow.pack(fill="x", pady=(6, 0))
        self.out_meta = self._text(out, "Compression applies on export", size=12, fg=MUTED)
        self.out_meta.pack(anchor="w", pady=(8, 0))
        self.out_size = self._text(out, "—", size=28, bold=True)
        self.out_size.pack(anchor="w")
        self.out_note = self._text(out, "Choose a video to estimate the output size.",
                                   size=12, fg=MUTED, wrap=420)
        self.out_note.pack(anchor="w", pady=(2, 0))
        self.size_bar = tk.Canvas(out, height=8, bg=CARD, highlightthickness=0, bd=0)
        self.size_bar.pack(fill="x", pady=(8, 0))
        self.size_bar.bind("<Configure>", lambda _e: self._paint_bar(self._bar_ratio, self._bar_warn))

        settings_outer, settings = self._card(body)
        settings_outer.pack(fill="x", padx=20, pady=(12, 0))
        self._text(settings, "DETECTION", size=11, fg=MUTED, bold=True).pack(anchor="w")
        cols = tk.Frame(settings, bg=CARD)
        cols.pack(fill="x", pady=(8, 0))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)
        cols.columnconfigure(2, weight=1)
        self._setting_column(cols, 0, "Face detection", "Higher ignores hands and objects.",
                             self.det_thr, 0.45, 0.90)
        self._setting_column(cols, 1, "Same-person strictness", "Higher splits similar people.",
                             self.strict, 0.30, 0.60)
        self._setting_column(cols, 2, "Blur margin", "Extra area around each face.",
                             self.margin, 0.10, 0.80)
        style_row = tk.Frame(settings, bg=CARD)
        style_row.pack(fill="x", pady=(10, 0))
        self._text(style_row, "Blur style", size=12, bold=True).pack(side="left", padx=(0, 10))
        styles = (("Blur", "Strong blur"), ("Pixelate", "Pixelate"), ("Box", "Black box"))
        srow, self.style_buttons = self._choice_buttons(style_row, styles, self.style.set)
        srow.pack(side="left")
        shape_row = tk.Frame(settings, bg=CARD)
        shape_row.pack(fill="x", pady=(8, 0))
        self._text(shape_row, "Blur shape", size=12, bold=True).pack(side="left", padx=(0, 10))
        shape_choices = (("Auto", "Auto"), ("Circle", "Circle"), ("Ellipse", "Ellipse"),
                         ("Rectangle", "Rectangle"))
        shrow, self.shape_buttons = self._choice_buttons(shape_row, shape_choices, self.shape.set)
        shrow.pack(side="left")
        self._text(settings, "Auto picks a circle or an ellipse. The edge fades into the picture.",
                   size=11, fg=MUTED).pack(anchor="w", pady=(6, 0))

        actions = tk.Frame(body, bg=BG)
        actions.pack(fill="x", padx=20, pady=(12, 0))
        self.btn_analyze = self._button(actions, "Analyze faces", self.start_analyze, primary=True)
        self.btn_analyze.pack(side="left")
        self.btn_regroup = self._button(actions, "Update groups", self.regroup)
        self.btn_regroup.pack(side="left", padx=(8, 0))
        self.btn_cancel = self._button(actions, "Cancel", self.cancel.set)
        self.btn_cancel.pack(side="right")

        people = tk.Frame(body, bg=BG)
        people.pack(fill="x", padx=20, pady=(14, 16))
        people_head = tk.Frame(people, bg=BG)
        people_head.pack(fill="x")
        self._text(people_head, "People", size=16, bold=True).pack(side="left")
        self.keep_count = self._text(people_head, "Analyze to choose who stays visible", size=12, fg=MUTED)
        self.keep_count.pack(side="right")
        ttk.Checkbutton(people, text="Hide appearances under 1 second. They stay blurred.",
                        variable=self.hide_brief, command=self.refresh_people).pack(anchor="w", pady=(4, 0))
        self.inner = tk.Frame(people, bg=BG)
        self.inner.pack(fill="x", pady=(6, 0))

        tk.Frame(footer, bg=LINE, height=1).pack(fill="x")
        foot = tk.Frame(footer, bg=BG)
        foot.pack(fill="x", padx=20, pady=12)
        self.status = self._text(foot, "Choose a video to begin.", size=12, fg=MUTED, wrap=980)
        self.status.pack(anchor="w")
        self.progress = ttk.Progressbar(foot, style="Accent.Horizontal.TProgressbar", maximum=1000)
        self.progress.pack(fill="x", pady=(8, 8))
        self.btn_export = self._button(foot, "Export blurred video", self.start_export, primary=True)
        self.btn_export.pack(fill="x")
        self._text(foot, "Video stays on this machine. Face models download once to ~/.cache/blurface.",
                   size=11, fg=MUTED).pack(anchor="w", pady=(8, 0))

        self.crf.trace_add("write", self._on_output_setting)
        self.max_edge.trace_add("write", self._on_output_setting)
        self.style.trace_add("write", lambda *_: self.sync_choices())
        self.shape.trace_add("write", lambda *_: self.sync_choices())
        self.out_var.trace_add("write", self._on_path_change)
        self.refresh_sizes()
        self.refresh_people()
        self.set_busy(False)
        r.bind("<Command-o>", self._shortcut_open)
        r.bind("<Control-o>", self._shortcut_open)
        r.bind("<Command-Return>", self._shortcut_export)
        r.bind("<Control-Return>", self._shortcut_export)
        self._install_mousewheel()
        r.protocol("WM_DELETE_WINDOW", self.on_close)
        self._sync_scrollregion()

    def _init_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TCheckbutton", background=BG, foreground=INK, font=("Helvetica", 12))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("Card.TCheckbutton", background=CARD, foreground=INK, font=("Helvetica", 12))
        style.map("Card.TCheckbutton", background=[("active", CARD)])
        style.configure("TEntry", fieldbackground="#fff", padding=6)
        style.configure("TScale", background=CARD, troughcolor=LINE)
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=LINE, background=ACCENT, bordercolor=BG,
                        lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        style.configure("Primary.TButton", background=ACCENT, foreground="white",
                        borderwidth=0, padding=(14, 8), font=("Helvetica", 13, "bold"),
                        focuscolor=ACCENT)
        style.map("Primary.TButton",
                  background=[("pressed", ACCENT_PRESS), ("active", ACCENT_PRESS), ("disabled", "#e7e2d9")],
                  foreground=[("disabled", "#a8a29e")])
        style.configure("Secondary.TButton", background=CARD, foreground=INK,
                        borderwidth=1, padding=(14, 8), font=("Helvetica", 13),
                        focuscolor=ACCENT)
        style.map("Secondary.TButton",
                  background=[("pressed", SOFT), ("active", SOFT), ("disabled", "#e7e2d9")],
                  foreground=[("disabled", "#a8a29e")])
        style.configure("Danger.TButton", background=CARD, foreground=WARN,
                        borderwidth=1, padding=(14, 8), font=("Helvetica", 13),
                        focuscolor=WARN)
        style.map("Danger.TButton",
                  background=[("pressed", "#f4e4dc"), ("active", "#f4e4dc"), ("disabled", "#e7e2d9")],
                  foreground=[("disabled", "#a8a29e")])
        style.configure("Choice.TButton", background=SOFT, foreground=INK,
                        borderwidth=0, padding=(10, 5), font=("Helvetica", 12),
                        focuscolor=ACCENT)
        style.map("Choice.TButton",
                  background=[("pressed", "#e4ddd2"), ("active", "#e4ddd2"), ("disabled", "#e7e2d9")],
                  foreground=[("disabled", "#a8a29e")])
        style.configure("ChoiceOn.TButton", background=ACCENT, foreground="white",
                        borderwidth=0, padding=(10, 5), font=("Helvetica", 12, "bold"),
                        focuscolor=ACCENT)
        style.map("ChoiceOn.TButton",
                  background=[("pressed", ACCENT_PRESS), ("active", ACCENT_PRESS), ("disabled", "#e7e2d9")],
                  foreground=[("disabled", "#a8a29e")])

    def _card(self, parent):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True, padx=14, pady=12)
        return outer, inner

    def _text(self, parent, text, size=13, fg=None, bg=None, bold=False, wrap=None):
        label = tk.Label(
            parent, text=text, fg=INK if fg is None else fg,
            bg=parent.cget("bg") if bg is None else bg,
            justify="left", anchor="w",
            font=("Helvetica", size, "bold" if bold else "normal"),
            wraplength=wrap or 0,
        )
        return label

    def _button(self, parent, text, command, primary=False):
        style_name = "Primary.TButton" if primary else "Secondary.TButton"
        return ttk.Button(parent, text=text, command=command, style=style_name, cursor="hand2")

    def _choice_buttons(self, parent, pairs, setter):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        buttons = []
        for label, value in pairs:
            btn = ttk.Button(row, text=label, command=lambda v=value: setter(v),
                             style="Choice.TButton", cursor="hand2")
            btn.pack(side="left", padx=(0, 6))
            buttons.append((value, btn))
        return row, buttons

    def _paint_choice(self, btn, selected):
        btn.config(style="ChoiceOn.TButton" if selected else "Choice.TButton")

    def _setting_column(self, parent, column, title, tip, var, lo, hi):
        col = tk.Frame(parent, bg=CARD)
        col.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 12, 0))
        top = tk.Frame(col, bg=CARD)
        top.pack(fill="x")
        self._text(top, title, size=12, bold=True).pack(side="left")
        value = self._text(top, "", size=12, fg=MUTED)
        value.pack(side="right")

        def upd(_v=None):
            value.config(text=f"{float(var.get()):.2f}")

        scale = ttk.Scale(col, from_=lo, to=hi, variable=var, command=upd)
        scale.pack(fill="x", pady=(4, 0))
        upd()
        self._text(col, tip, size=11, fg=MUTED, wrap=240).pack(anchor="w", pady=(2, 0))
        self.setting_scales.append(scale)

    def _paint_bar(self, ratio, warn):
        self._bar_ratio = max(0.0, float(ratio or 0))
        self._bar_warn = bool(warn)
        canvas = self.size_bar
        width = canvas.winfo_width()
        height = max(canvas.winfo_height(), 8)
        if width < 4:
            return
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=LINE, width=0)
        if self._bar_ratio <= 0:
            return
        fill = WARN if warn else ACCENT
        canvas.create_rectangle(0, 0, max(3, int(width * min(self._bar_ratio, 1.0))),
                                height, fill=fill, width=0)

    def _fit_page_width(self, event):
        if event.width < 2 or event.width == self._page_width:
            return
        self._page_width = event.width
        self.canvas.itemconfigure(self.win, width=event.width)
        self._sync_scrollregion()

    def _sync_scrollregion(self, _event=None):
        try:
            if not self.canvas.winfo_exists():
                return
            self.page.update_idletasks()
            width = max(self.canvas.winfo_width(), 1)
            height = max(self.page.winfo_reqheight(), 1)
            self.canvas.configure(scrollregion=(0, 0, width, height))
        except tk.TclError:
            return

    def _install_mousewheel(self):
        """Scroll with a mouse wheel or a macOS trackpad.

        Tk 9 sends trackpad movement as <TouchpadScroll>, not <MouseWheel>.
        Python also turns fractional wheel deltas into 0, so both bindings stay
        in Tcl. yscrollincrement is one pixel.
        """
        self.canvas.configure(yscrollincrement=1)
        name = self.canvas._w
        self.root.tk.call("set", "::blurface_wheel", "0.0")
        wheel = (
            "set raw %D; "
            "if {abs($raw) < 20} { set raw [expr {$raw * 16.0}] }; "
            "set ::blurface_wheel [expr {$::blurface_wheel + (-1.0 * $raw * [tk scaling] * 0.75)}]; "
            "set step [expr {int($::blurface_wheel)}]; "
            "set ::blurface_wheel [expr {$::blurface_wheel - $step}]; "
            f"if {{$step != 0}} {{ {name} yview scroll $step units }}"
        )
        touch = (
            "lassign [tk::PreciseScrollDeltas %D] ::blurface_dx ::blurface_dy; "
            f"if {{$::blurface_dy != 0}} {{ {name} yview scroll [expr {{-int($::blurface_dy)}}] units }}"
        )
        self.root.tk.call("bind", "all", "<MouseWheel>", wheel)
        # The installers ship Tk 8.6, which has no TouchpadScroll event.
        # Binding it there aborts the app on launch. Tk 9 uses it for trackpads.
        if self.root.tk.call("info", "commands", "::tk::PreciseScrollDeltas"):
            try:
                self.root.tk.call("bind", "all", "<TouchpadScroll>", touch)
            except tk.TclError:
                pass
        self.root.tk.call("bind", "all", "<Button-4>", f"{name} yview scroll -48 units")
        self.root.tk.call("bind", "all", "<Button-5>", f"{name} yview scroll 48 units")

    def _wheel(self, event):
        steps = wheel_steps(getattr(event, "delta", 0), getattr(event, "num", 0))
        if steps and self.canvas.cget("yscrollincrement"):
            steps *= 16
        if steps:
            self.canvas.yview_scroll(steps, "units")
        return "break"

    def _shortcut_open(self, _event=None):
        if not self.busy:
            self.open_video()
        return "break"

    def _shortcut_export(self, _event=None):
        if not self.busy:
            self.start_export()
        return "break"

    def on_close(self):
        self.cancel.set()
        for sequence in ("<MouseWheel>", "<TouchpadScroll>", "<Button-4>", "<Button-5>"):
            try:
                self.root.unbind_all(sequence)
            except tk.TclError:
                pass
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self.root.destroy()

    def _schedule(self):
        try:
            if self.root.winfo_exists():
                self._after_id = self.root.after(100, self._poll)
        except tk.TclError:
            pass

    def _enable(self, btn, enabled, primary=False):
        btn.config(state="normal" if enabled else "disabled",
                   style="Primary.TButton" if primary else "Secondary.TButton")

    def set_busy(self, busy):
        self.busy = busy
        has_video = bool(self.src)
        has_analysis = self.analysis is not None
        self._enable(self.btn_open, not busy, primary=True)
        self._enable(self.btn_analyze, (not busy) and has_video, primary=True)
        self._enable(self.btn_browse, not busy, primary=False)
        self._enable(self.btn_regroup, (not busy) and has_analysis, primary=False)
        self._enable(self.btn_export, (not busy) and has_analysis, primary=True)
        self.btn_cancel.config(state="normal" if busy else "disabled", style="Danger.TButton")
        scale_state = "disabled" if busy else "normal"
        self.quality_scale.config(state=scale_state)
        self.out_entry.config(state=scale_state)
        for scale in self.setting_scales:
            scale.config(state=scale_state)
        self.sync_choices()

    def sync_choices(self):
        groups = (
            (self.quality_buttons, int(round(float(self.crf.get())))),
            (self.resolution_buttons, int(self.max_edge.get())),
            (self.style_buttons, self.style.get()),
            (self.shape_buttons, self.shape.get()),
        )
        for buttons, current in groups:
            for value, btn in buttons:
                self._paint_choice(btn, value == current)
                btn.config(state="disabled" if self.busy else "normal")

    def say(self, text):
        self.status.config(text=text)

    def _on_output_setting(self, *_args):
        self.output_actual = None
        self.refresh_sizes()

    def _on_path_change(self, *_args):
        self.output_actual = None
        self.refresh_sizes()

    def _planned_output(self):
        if not self.meta:
            return None
        meta = self.meta
        crf = int(round(float(self.crf.get())))
        edge = int(self.max_edge.get())
        estimate = estimate_output_bytes(
            meta["width"], meta["height"], meta["frames"] or 1, meta["fps"], crf, edge)
        width, height = fit_dimensions(meta["width"], meta["height"], edge)
        return estimate, width, height, crf

    def refresh_sizes(self, *_args):
        if not hasattr(self, "in_size") or not self.root.winfo_exists():
            return
        try:
            meta = self.meta
            crf_now = int(round(float(self.crf.get())))
            self.quality_caption.config(text=f"{quality_name(crf_now)} · CRF {crf_now}")
            if not meta:
                self.in_name.config(text="No video yet")
                self.in_path.config(text="Choose a video to see its size.")
                self.in_meta.config(text="Resolution, frame rate, and duration")
                self.in_size.config(text="—")
                self.out_kind.config(text="Estimated")
                self.out_meta.config(text="Compression applies on export")
                self.out_size.config(text="—")
                self.out_note.config(text="Choose a video to estimate the output size.")
                self._paint_bar(0, False)
                self.sync_choices()
                return
            plan = self._planned_output()
            estimate, width, height, crf = plan
            self.in_name.config(text=shorten_path(os.path.basename(meta["path"]), 42))
            self.in_path.config(text=shorten_path(meta["path"], 64))
            self.in_meta.config(text=(
                f"{meta['width']}×{meta['height']} · {format_fps(meta['fps'])} · "
                f"{timecode(meta['duration'])}"
            ))
            self.in_size.config(text=format_size(meta["bytes"]))
            self.quality_caption.config(text=f"{quality_name(crf)} · CRF {crf}")
            self.out_meta.config(text=f"Exports at {width}×{height}")
            actual = self.output_actual
            saved = actual is not None
            shown = actual if saved else estimate
            self.out_kind.config(text="Saved" if saved else "Estimated")
            self.out_size.config(text=format_size(shown) if saved else "~" + format_size(shown))
            self.out_note.config(text=comparison_sentence(shown, meta["bytes"], estimated=not saved))
            ratio = (shown / meta["bytes"]) if meta["bytes"] else 0
            self._paint_bar(ratio, ratio >= 1.05)
            self.sync_choices()
        except tk.TclError:
            return

    def open_video(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a video",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("All", "*.*")])
        if not path:
            return
        meta = probe_video(path)
        if meta is None:
            messagebox.showerror("Blurface", "Could not read that video.", parent=self.root)
            return
        self.src = path
        self.meta = meta
        self.analysis = None
        self.persons = []
        self.keep_vars = {}
        self.output_actual = None
        base, _ext = os.path.splitext(path)
        self.out_var.set(base + "_blurred.mp4")
        self.progress["value"] = 0
        self.refresh_people()
        self.refresh_sizes()
        self.say("Analyze the video, then tick only the people who should stay visible.")
        self.set_busy(False)

    def pick_output(self):
        if self.busy:
            return
        current = self.out_var.get().strip()
        initialdir = os.path.dirname(current) or None
        initialfile = os.path.basename(current) if current else "blurred.mp4"
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save blurred video", defaultextension=".mp4",
            initialdir=initialdir, initialfile=initialfile, filetypes=[("MP4", "*.mp4")])
        if path:
            self.out_var.set(path)

    def start_analyze(self):
        if self.busy:
            return
        if not self.src:
            messagebox.showinfo("Blurface", "Choose a video first.", parent=self.root)
            return
        self.cancel.clear()
        self.set_busy(True)
        self.progress["value"] = 0
        src, thr = self.src, float(self.det_thr.get())

        def work():
            try:
                engine = FaceEngine(thr, status=lambda s: self.q.put(("status", s)))
                ana = analyze_video(src, engine,
                                    lambda frac, text: self.q.put(("progress", frac, text)),
                                    self.cancel)
                self.q.put(("cancelled",) if self.cancel.is_set() else ("analysis_done", ana))
            except Exception as exc:
                traceback.print_exc()
                self.q.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def regroup(self):
        if not self.analysis or self.busy:
            return
        self.keep_vars = {}
        self.persons = cluster_tracks(self.analysis.tracks, float(self.strict.get()))
        self.refresh_people()
        self._summary()

    def _summary(self):
        analysis = self.analysis
        unclear = sum(1 for track in analysis.tracks if track.person is None)
        text = f"{len(self.persons)} people from {len(analysis.tracks)} face tracks. "
        if unclear:
            text += f"{unclear} unclear tracks stay blurred. "
        text += "Tick who stays visible, then export."
        self.say(text)

    def update_keep_count(self):
        if not self.analysis:
            self.keep_count.config(text="Analyze to choose who stays visible")
            return
        kept = sum(1 for var in self.keep_vars.values() if var.get())
        self.keep_count.config(text=f"{kept} kept visible · unchecked people are blurred")

    def refresh_people(self):
        try:
            self._refresh_people()
        finally:
            if hasattr(self, "canvas"):
                self._sync_scrollregion()

    def _refresh_people(self):
        for child in self.inner.winfo_children():
            child.destroy()
        self.thumb_refs.clear()
        if not self.analysis:
            tk.Label(self.inner, bg=BG, fg=MUTED, justify="left", font=("Helvetica", 13),
                     text="Analyze a video to list the people in it.\n"
                          "Leave someone unchecked and their face is blurred for the whole clip."
                     ).pack(anchor="w", padx=4, pady=12)
            self.update_keep_count()
            return
        fps = self.analysis.fps
        visible = [p for p in self.persons if not (self.hide_brief.get() and p.frames < fps)]
        if not visible:
            tk.Label(self.inner, bg=BG, fg=MUTED, font=("Helvetica", 13),
                     text="No faces found. Lower the detection threshold and analyze again."
                     ).pack(anchor="w", padx=4, pady=12)
            self.update_keep_count()
            return
        for person in visible:
            card = tk.Frame(self.inner, bg=CARD, highlightbackground=LINE, highlightthickness=1)
            card.pack(fill="x", padx=2, pady=4)
            var = self.keep_vars.setdefault(person.id, tk.BooleanVar(value=False))
            ttk.Checkbutton(card, text="Keep visible", variable=var, style="Card.TCheckbutton",
                            command=self.update_keep_count).pack(side="right", padx=12, pady=8)
            for crop in person.thumbs(3):
                ok, buf = cv2.imencode(".png", crop)
                if not ok:
                    continue
                img = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
                self.thumb_refs.append(img)
                tk.Label(card, image=img, bg=CARD).pack(side="left", padx=(8, 2), pady=8)
            tk.Label(card, bg=CARD, fg=INK, justify="left", font=("Helvetica", 13),
                     text=f"Person {person.id}\n{person.frames / fps:.0f}s on screen · "
                          f"first seen {timecode(person.first_frame / fps)}"
                     ).pack(side="left", padx=10)
        self.update_keep_count()

    def start_export(self):
        if self.busy or not self.analysis:
            return
        if not self.analysis.n_frames or not self.analysis.size:
            messagebox.showerror("Blurface", "This video has no readable frames.", parent=self.root)
            return
        dst = self.out_var.get().strip()
        if not dst:
            messagebox.showwarning("Blurface", "Choose where to save the blurred video.", parent=self.root)
            return
        _base, ext = os.path.splitext(dst)
        if ext.lower() not in {".mp4", ".mov", ".m4v"}:
            dst = dst + ".mp4"
            self.out_var.set(dst)
        if os.path.abspath(dst) == os.path.abspath(self.src):
            messagebox.showerror("Blurface", "Choose a different file for the output.", parent=self.root)
            return
        if os.path.exists(dst):
            existing = format_size(os.path.getsize(dst))
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"{os.path.basename(dst)} already exists ({existing}). Overwrite it?",
                    parent=self.root):
                return
        keep = {pid for pid, var in self.keep_vars.items() if var.get()}
        if not keep and not messagebox.askyesno(
                "Blur everyone?",
                "No one is marked to keep, so every face will be blurred. Continue?",
                parent=self.root):
            return
        self.cancel.clear()
        self.set_busy(True)
        self.progress["value"] = 0
        ana = self.analysis
        style, pad = self.style.get(), float(self.margin.get())
        shape = self.shape.get()
        crf, max_edge = int(round(float(self.crf.get()))), int(self.max_edge.get())

        def work():
            try:
                self.q.put(("status", "Planning blur regions ..."))
                per_frame = compute_blur_boxes(ana, keep)
                res = render_video(
                    ana.path, dst, per_frame, style, pad, ana.fps,
                    lambda frac, text: self.q.put(("progress", frac, text)),
                    self.cancel, ana.n_frames, crf=crf, max_edge=max_edge, shape=shape)
                self.q.put(("cancelled",) if res is None else ("export_done", dst, res))
            except Exception as exc:
                traceback.print_exc()
                self.q.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _export_message(self, dst, res):
        in_bytes = self.meta["bytes"] if self.meta else None
        out_bytes = res.get("output_bytes")
        lines = [
            dst, "",
            f"Input    {format_size(in_bytes)}",
            f"Output   {format_size(out_bytes)}",
            comparison_sentence(out_bytes, in_bytes, estimated=False),
            "",
            f"{res['frames']} frames · blur on {res['blurred_frames']}",
            res["audio"], "",
            "Scrub through the result before you share it.",
        ]
        if res.get("ffmpeg_error"):
            lines.extend(["", "ffmpeg said:", res["ffmpeg_error"]])
        return "\n".join(lines)

    def _poll(self):
        try:
            if not self.root.winfo_exists():
                return
            while True:
                message = self.q.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self.progress["value"] = max(0, min(1000, int(float(message[1]) * 1000)))
                    self.say(message[2])
                elif kind == "status":
                    self.say(message[1])
                elif kind == "analysis_done":
                    self.analysis = message[1]
                    if self.meta and self.analysis.size:
                        width, height = self.analysis.size
                        self.meta["width"] = width
                        self.meta["height"] = height
                        self.meta["frames"] = self.analysis.n_frames
                        self.meta["fps"] = self.analysis.fps
                        self.meta["duration"] = (
                            self.analysis.n_frames / self.analysis.fps if self.analysis.fps else 0)
                    self.output_actual = None
                    self.progress["value"] = 1000
                    self.set_busy(False)
                    self.regroup()
                    self.refresh_sizes()
                elif kind == "export_done":
                    _dst, res = message[1], message[2]
                    self.output_actual = res.get("output_bytes")
                    if self.output_actual is None and os.path.isfile(message[1]):
                        self.output_actual = os.path.getsize(message[1])
                    self.set_busy(False)
                    self.refresh_sizes()
                    self.progress["value"] = 1000
                    self.say(f"Saved {format_size(self.output_actual)}")
                    messagebox.showinfo("Export finished", self._export_message(message[1], res),
                                        parent=self.root)
                elif kind == "cancelled":
                    self.set_busy(False)
                    self.say("Cancelled.")
                elif kind == "error":
                    self.set_busy(False)
                    self.say(message[1].splitlines()[0][:160])
                    messagebox.showerror("Blurface", message[1], parent=self.root)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self._schedule()


def main():
    if tk is None:
        sys.exit("Tkinter is not available. On macOS with Homebrew Python run: "
                 "brew install python-tk@3.12 (match your Python version).")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()