"""
detector.py - Robust Multi-pass Object Detection using Faster R-CNN (MobileNet V3)
on COCO dataset with strong false-positive filtering.

Pipeline:
  Pass 1: Full-image scan (high-confidence anchors)
  Pass 2: Tile-based detection (9 overlapping zones for missed objects)
  Pass 3: 2x upscale for small objects
  Pass 4: Contrast-enhanced scan for dark / low-visibility objects
  + Cross-pass confidence voting (require multi-pass agreement for low scores)
  + Strict NMS deduplication
  + Post-processing filters (min area, aspect ratio, edge clipping, score floor)
"""

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import numpy as np
import math
import io
import base64
import threading

# ─── TUNING PARAMETERS ────────────────────────────────────────────────────────
# Single-pass detection thresholds (model raw score gate)
PASS1_THRESHOLD = 0.40       # Full-image pass — primary anchor pass, keep reasonably high
PASS2_THRESHOLD = 0.40       # Tile pass
PASS3_THRESHOLD = 0.40       # Upscale pass
PASS4A_THRESHOLD = 0.35      # Contrast-enhanced pass A
PASS4B_THRESHOLD = 0.30      # Contrast-enhanced pass B (more aggressive enhancement)

# Post-merge confidence floor: any detection below this is dropped
FINAL_SCORE_FLOOR = 0.35

# Cross-pass voting: detections seen in only 1 pass must exceed this score
SINGLE_PASS_MIN_SCORE = 0.55

# Minimum bounding-box area as fraction of image area (filters tiny noise)
MIN_AREA_FRACTION = 0.0008   # 0.08 % of image

# Maximum bounding-box area as fraction of image area (filters full-frame ghosts)
MAX_AREA_FRACTION = 0.85     # 85 % of image

# Aspect ratio limits — reject boxes thinner than 1:8 or wider than 8:1
MAX_ASPECT_RATIO = 8.0

# Edge-margin: boxes whose center is within this fraction of image edge are
# penalised (lowered score) because edge crops produce many false positives
EDGE_MARGIN_FRACTION = 0.04  # 4 % from each edge

# NMS IoU thresholds
NMS_SAME_CLASS_IOU = 0.30    # Same class overlap to suppress
NMS_CROSS_CLASS_IOU = 0.70   # Different class overlap to suppress
NMS_CENTER_DIST_RATIO = 0.45 # Center-distance / avg-dim ratio for same-class suppression


# COCO 91-class label map used by torchvision detection models
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ObjectDetector:
    """Robust multi-pass COCO object detector using Faster R-CNN MobileNet V3."""

    def __init__(self):
        self.model = None
        self.device = None
        self.ready = False
        self.loading = True
        self.error = None

    def load_model(self):
        """Load the Faster R-CNN MobileNet V3 model (COCO trained)."""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Handle both old and new torchvision weight APIs
            try:
                from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
                self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                )
            except (ImportError, AttributeError):
                self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                    pretrained=True
                )

            self.model.to(self.device)
            self.model.eval()
            self.ready = True
            self.loading = False
            print("[detector] ✓ Faster R-CNN MobileNet V3 (COCO) loaded successfully")
            print(f"[detector]   Device: {self.device}")
        except Exception as e:
            self.error = str(e)
            self.loading = False
            print(f"[detector] ✗ Model load failed: {e}")

    def load_model_async(self):
        """Load model in a background thread."""
        thread = threading.Thread(target=self.load_model)
        thread.daemon = True
        thread.start()

    def _detect_single(self, pil_image, threshold=0.40):
        """Run detection on a single PIL image and return results above threshold."""
        if not self.ready:
            return []

        tensor = F.to_tensor(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor)[0]

        results = []
        for i in range(len(predictions['scores'])):
            score = predictions['scores'][i].item()
            if score < threshold:
                continue

            label_idx = predictions['labels'][i].item()
            if label_idx >= len(COCO_CLASSES):
                continue

            class_name = COCO_CLASSES[label_idx]
            if class_name in ('__background__', 'N/A'):
                continue

            box = predictions['boxes'][i].cpu().numpy()
            # Convert from [x1, y1, x2, y2] to [x, y, w, h] format
            x1, y1, x2, y2 = box
            w, h = float(x2 - x1), float(y2 - y1)

            # Skip degenerate boxes
            if w < 2 or h < 2:
                continue

            results.append({
                'class': class_name,
                'score': round(float(score), 4),
                'bbox': [float(x1), float(y1), w, h]
            })

        return results

    # ── GEOMETRY FILTERS ───────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_area(preds, img_w, img_h):
        """Remove boxes that are too small (noise) or too large (full-frame ghosts)."""
        img_area = img_w * img_h
        min_area = img_area * MIN_AREA_FRACTION
        max_area = img_area * MAX_AREA_FRACTION
        return [p for p in preds if min_area <= p['bbox'][2] * p['bbox'][3] <= max_area]

    @staticmethod
    def _filter_by_aspect_ratio(preds):
        """Remove boxes with extreme aspect ratios (thin slivers = noise)."""
        out = []
        for p in preds:
            w, h = p['bbox'][2], p['bbox'][3]
            if w < 1 or h < 1:
                continue
            ratio = max(w / h, h / w)
            if ratio <= MAX_ASPECT_RATIO:
                out.append(p)
        return out

    @staticmethod
    def _penalise_edge_detections(preds, img_w, img_h):
        """Lower confidence of detections whose centers are very close to the image edge."""
        mx = img_w * EDGE_MARGIN_FRACTION
        my = img_h * EDGE_MARGIN_FRACTION
        for p in preds:
            cx = p['bbox'][0] + p['bbox'][2] / 2
            cy = p['bbox'][1] + p['bbox'][3] / 2
            if cx < mx or cx > img_w - mx or cy < my or cy > img_h - my:
                p['score'] = round(p['score'] * 0.7, 4)  # 30 % penalty
        return preds

    @staticmethod
    def _clip_boxes(preds, img_w, img_h):
        """Clip bounding boxes to image boundaries."""
        for p in preds:
            x, y, w, h = p['bbox']
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            p['bbox'] = [x, y, w, h]
        return preds

    # ── CROSS-PASS CONFIDENCE VOTING ─────────────────────────────────────────

    @staticmethod
    def _cross_pass_vote(all_preds_with_pass):
        """
        Require low-confidence detections to be confirmed by multiple passes.
        Each pred has an extra '_pass' key indicating which pass found it.
        If a class+region is only seen in 1 pass and score < SINGLE_PASS_MIN_SCORE,
        it is dropped.
        """
        # Build a quick spatial index: for each prediction, count how many passes
        # produced a similar detection (same class, IoU > 0.20 with any other pass)
        for i, p in enumerate(all_preds_with_pass):
            confirming_passes = set()
            confirming_passes.add(p['_pass'])
            for j, q in enumerate(all_preds_with_pass):
                if i == j:
                    continue
                if q['class'] != p['class']:
                    continue
                if q['_pass'] == p['_pass']:
                    continue
                if ObjectDetector._iou(p['bbox'], q['bbox']) > 0.20:
                    confirming_passes.add(q['_pass'])
            p['_num_passes'] = len(confirming_passes)

        # Filter: keep if high-confidence OR confirmed by ≥2 passes
        kept = []
        for p in all_preds_with_pass:
            if p['score'] >= SINGLE_PASS_MIN_SCORE or p['_num_passes'] >= 2:
                kept.append(p)

        # Clean up helper keys
        for p in kept:
            p.pop('_pass', None)
            p.pop('_num_passes', None)
        return kept

    # ── MAIN DETECTION PIPELINE ────────────────────────────────────────────────

    def detect_multipass(self, image_data_url):
        """
        Robust multi-pass detection pipeline:
          Pass 1: Full-image scan
          Pass 2: 9 overlapping tile scan
          Pass 3: 2x upscale for small objects
          Pass 4: Contrast-enhanced scan (two variants)
          + Cross-pass voting  (reject single-pass low-confidence ghosts)
          + Strict NMS
          + Geometry filters   (area, aspect ratio, edge penalty)
          + Final score floor
        Returns: (detections_list, image_width, image_height)
        """
        # Decode base64 image
        if ',' in image_data_url:
            image_data_url = image_data_url.split(',')[1]

        image_bytes = base64.b64decode(image_data_url)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        W, H = pil_image.size

        # ── PASS 1: Full image ─────────────────────────────────────────────────
        raw_full = self._detect_single(pil_image, threshold=PASS1_THRESHOLD)
        for p in raw_full:
            p['_pass'] = 1

        # ── PASS 2: Tile-based detection (9 overlapping tiles, 3x3 grid) ──────
        raw_tile = []
        cols, rows = 3, 3
        tw, th = int(W * 0.55), int(H * 0.55)
        for r in range(rows):
            for c in range(cols):
                sx = min(int(c * (W - tw) / max(cols - 1, 1)), W - tw)
                sy = min(int(r * (H - th) / max(rows - 1, 1)), H - th)
                tile = pil_image.crop((sx, sy, sx + tw, sy + th))
                tile_preds = self._detect_single(tile, threshold=PASS2_THRESHOLD)
                for p in tile_preds:
                    p['bbox'] = [
                        p['bbox'][0] + sx,
                        p['bbox'][1] + sy,
                        p['bbox'][2],
                        p['bbox'][3]
                    ]
                    p['_pass'] = 2
                raw_tile.extend(tile_preds)

        # ── PASS 3: 2x upscale for small objects ──────────────────────────────
        scale = 2.0
        scaled_image = pil_image.resize(
            (int(W * scale), int(H * scale)), Image.LANCZOS
        )
        raw_scaled = self._detect_single(scaled_image, threshold=PASS3_THRESHOLD)
        scaled_mapped = []
        for p in raw_scaled:
            scaled_mapped.append({
                'class': p['class'],
                'score': p['score'],
                'bbox': [
                    p['bbox'][0] / scale,
                    p['bbox'][1] / scale,
                    p['bbox'][2] / scale,
                    p['bbox'][3] / scale
                ],
                '_pass': 3
            })

        # ── PASS 4: Contrast-enhanced scan ─────────────────────────────────────
        # 4A: contrast(160%) brightness(115%) color(140%)
        enh1 = ImageEnhance.Contrast(pil_image).enhance(1.6)
        enh1 = ImageEnhance.Brightness(enh1).enhance(1.15)
        enh1 = ImageEnhance.Color(enh1).enhance(1.4)
        raw_enh1 = self._detect_single(enh1, threshold=PASS4A_THRESHOLD)
        for p in raw_enh1:
            p['_pass'] = 4

        # 4B: contrast(200%) brightness(140%) for darker objects
        enh2 = ImageEnhance.Contrast(pil_image).enhance(2.0)
        enh2 = ImageEnhance.Brightness(enh2).enhance(1.4)
        raw_enh2 = self._detect_single(enh2, threshold=PASS4B_THRESHOLD)
        for p in raw_enh2:
            p['_pass'] = 4

        raw_enh_all = raw_enh1 + raw_enh2

        # ── CROSS-PASS VOTING ──────────────────────────────────────────────────
        # Gather all raw detections and require multi-pass agreement for low scores
        all_raw = raw_full + raw_tile + scaled_mapped + raw_enh_all
        voted = self._cross_pass_vote(all_raw)

        # ── CLIP BOXES to image boundaries ─────────────────────────────────────
        voted = self._clip_boxes(voted, W, H)

        # ── GEOMETRY FILTERS ───────────────────────────────────────────────────
        voted = self._filter_by_area(voted, W, H)
        voted = self._filter_by_aspect_ratio(voted)
        voted = self._penalise_edge_detections(voted, W, H)

        # ── STRICT NMS ─────────────────────────────────────────────────────────
        merged = self._strict_nms(voted)

        # ── FINAL SCORE FLOOR ──────────────────────────────────────────────────
        merged = [p for p in merged if p['score'] >= FINAL_SCORE_FLOOR]

        print(f"[detector] Detection complete: {len(merged)} objects from {len(all_raw)} raw candidates")
        return merged, W, H

    def _strict_nms(self, preds):
        """
        Strict NMS per class with tight IoU.
        - Same class: drop if IoU > NMS_SAME_CLASS_IOU  OR  centers within
          NMS_CENTER_DIST_RATIO of avg dimension
        - Cross-class: drop if IoU > NMS_CROSS_CLASS_IOU
        """
        preds.sort(key=lambda x: x['score'], reverse=True)
        kept = []

        for pred in preds:
            drop = False
            for k in kept:
                overlap = self._iou(k['bbox'], pred['bbox'])

                # Same class: drop if significant overlap
                if k['class'] == pred['class'] and overlap > NMS_SAME_CLASS_IOU:
                    drop = True
                    break

                # Cross-class: drop if highly overlapping (same region, different label)
                if overlap > NMS_CROSS_CLASS_IOU:
                    drop = True
                    break

                # Same class, center-distance check (catches shifted duplicates)
                if k['class'] == pred['class']:
                    ax, ay, aw, ah = k['bbox']
                    bx, by, bw, bh = pred['bbox']
                    dx = (ax + aw / 2) - (bx + bw / 2)
                    dy = (ay + ah / 2) - (by + bh / 2)
                    dist = math.sqrt(dx * dx + dy * dy)
                    avg_dim = (math.sqrt(aw * ah) + math.sqrt(bw * bh)) / 2
                    if avg_dim > 0 and dist < avg_dim * NMS_CENTER_DIST_RATIO:
                        drop = True
                        break

            if not drop:
                kept.append(pred)

        return kept

    @staticmethod
    def _iou(a, b):
        """Calculate IoU between two [x, y, w, h] boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx)
        iy = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix2 <= ix or iy2 <= iy:
            return 0
        inter = (ix2 - ix) * (iy2 - iy)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0
