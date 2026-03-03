# VLA-4 :: Chain-of-Thought Safety Verifier

A lightweight **Vision–Language–Action (VLA)** safety verification system that combines real-time object detection with Chain-of-Thought (CoT) reasoning to ensure safe robot action execution. The system captures a live camera feed, detects objects using a multi-pass deep learning pipeline, and verifies operator commands against safety and grounding policies before generating a structured robotic action plan.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Safety Verification Pipeline](#safety-verification-pipeline)
- [Object Detection Pipeline](#object-detection-pipeline)
- [Chain-of-Thought Engine](#chain-of-thought-engine)
- [Configuration & Tuning](#configuration--tuning)
- [References](#references)

---

## Overview

Modern Vision-Language-Action (VLA) models can translate visual observations and natural language commands into robotic actions. However, they can be susceptible to:

1. **Safety violations** — executing commands that may cause physical harm to humans.
2. **Hallucinations** — acting on objects that are not actually present in the scene.
3. **Ungrounded behaviour** — operating without a clear, verifiable reasoning chain.

This project implements a **CoT Safety Verifier** that sits between the operator's command and the robot's execution layer. It:

- Detects all objects in the robot's visual field using a robust multi-pass detector.
- Screens every command against safety harm patterns and dangerous object–human combinations.
- Cross-references mentioned objects against the detected scene to catch hallucinations.
- Generates a structured, human-readable Chain-of-Thought action plan before execution is permitted.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Pass Object Detection** | 4-pass detection pipeline (full image, tiled, upscaled, contrast-enhanced) using Faster R-CNN MobileNet V3 trained on 80 COCO classes |
| **Cross-Pass Voting** | Low-confidence detections must be confirmed by ≥ 2 passes to survive filtering |
| **Strict NMS** | Same-class IoU, cross-class IoU, and center-distance suppression to eliminate duplicates |
| **Geometry Filters** | Area bounds, aspect ratio limits, and edge-margin penalties remove false positives |
| **Safety Verification** | Regex-based harm pattern matching (physical harm, self-harm, weapon misuse) |
| **Dangerous Combo Detection** | Blocks actions when sharp objects + humans are detected together with harmful intent |
| **Hallucination Detection** | Cross-references command nouns against the detected object inventory |
| **Structured CoT Generation** | Phase-based action plans with preconditions, numbered phases, and measurable parameters |
| **Hybrid Detection (Optional)** | Claude Vision API integration for detecting objects beyond COCO's 80 classes |
| **LLM-Powered CoT (Optional)** | Claude API for generating higher-quality, context-aware reasoning chains |
| **Real-Time Camera Feed** | Browser-based camera capture with crosshair overlay and bounding-box visualization |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Browser Frontend                       │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Camera   │  │ Object List  │  │  CoT Output Panel  │  │
│  │  Feed +   │  │ (detected    │  │  (safety status,   │  │
│  │  Canvas   │  │  objects)    │  │   action phases)   │  │
│  └─────┬────┘  └──────┬───────┘  └─────────┬──────────┘  │
│        │               │                    │             │
│        └───────────────┼────────────────────┘             │
│                        │ REST API (JSON)                  │
└────────────────────────┼──────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                  Flask Backend (Python)                     │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ detector.py  │  │  safety.py   │  │  cot_engine.py   │  │
│  │             │  │              │  │                  │  │
│  │ Faster RCNN │  │ Harm Pattern │  │ Demo CoT         │  │
│  │ MobileNetV3 │  │ Matching     │  │ (phase-based)    │  │
│  │ Multi-Pass  │  │              │  │                  │  │
│  │ Detection   │  │ Dangerous    │  │ LLM CoT          │  │
│  │             │  │ Combo Check  │  │ (Claude API)     │  │
│  │ Cross-Pass  │  │              │  │                  │  │
│  │ Voting +    │  │ Hallucination│  │ Claude Vision    │  │
│  │ Strict NMS  │  │ Grounding    │  │ Detection        │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│                                                            │
│                     app.py (Flask Server)                   │
│              Routes: / , /api/status, /api/detect,         │
│                          /api/verify                       │
└────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AIS-Project/
├── app.py                  # Flask web server, REST API endpoints
├── detector.py             # Multi-pass object detection (Faster R-CNN)
├── safety.py               # Safety verification & hallucination detection
├── cot_engine.py           # Chain-of-Thought generation engine
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI (HTML + CSS + JavaScript)
├── vla_cot_verifier.html   # Original standalone HTML reference
├── References/
│   ├── CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models.pdf
│   ├── Robotic Control via Embodied Chain-of-Thought Reasoning.pdf
│   ├── SafeVLA.pdf
│   └── VLSA.pdf
└── README.md
```

### Module Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~274 | Flask web server with 4 routes. Orchestrates detection → safety check → hallucination check → CoT generation pipeline. |
| `detector.py` | ~409 | `ObjectDetector` class implementing 4-pass detection, cross-pass voting, strict NMS, and geometry-based post-processing filters. |
| `safety.py` | ~108 | `safety_check()` for harm pattern matching & dangerous combos; `hallucination_check()` for grounding verification. |
| `cot_engine.py` | ~388 | Phase-based demo CoT generator for 6 action types. Optional Claude API integration for LLM CoT and hybrid vision detection. |
| `templates/index.html` | ~1191 | Full frontend with camera feed, object list panel, CoT output panel, and API key input. Uses Exo 2 + Share Tech Mono fonts. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, Flask 3.x |
| **Object Detection** | PyTorch, torchvision — Faster R-CNN MobileNet V3 Large FPN (COCO-trained, 80 classes) |
| **Frontend** | Vanilla HTML5, CSS3 (custom dark theme), JavaScript (Fetch API) |
| **Optional LLM** | Anthropic Claude API (claude-haiku-4-5-20251001) for enhanced CoT and vision detection |
| **Image Processing** | Pillow (PIL), NumPy |

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip package manager
- A webcam (for live camera features)
- (Optional) Anthropic API key for Claude-powered features

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd AIS-Project

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download the Faster R-CNN MobileNet V3 model weights (~60 MB) automatically from PyTorch Hub.

---

## Usage

### Starting the Server

```bash
python app.py
```

The server starts at `http://localhost:5000`. The console will show:

```
============================================================
  VLA-4 :: CoT Safety Verifier — Python Backend
  Loading Faster R-CNN MobileNet V3 (COCO) model...
============================================================
[detector] ✓ Faster R-CNN MobileNet V3 (COCO) loaded successfully
[detector]   Device: cpu
```

### Using the Web Interface

1. **Open** `http://localhost:5000` in a modern browser (Chrome/Edge recommended).
2. **Start Camera** — click the button to enable the webcam feed.
3. **Capture** — freeze a frame from the live video.
4. **Detect Objects** — runs the multi-pass detection pipeline on the captured image.
5. **Enter a Command** — type a robot action command (e.g., `take the knife and cut the apple`).
6. **Run CoT** — submits the command for safety verification and CoT generation.

### Quick Test Scenarios

The UI provides three preset scenario buttons:

| Button | Command | Expected Outcome |
|--------|---------|------------------|
| **S-1 Valid** | `take the knife and cut the apple` |  Success — generates full action plan |
| **S-2 Hallucination** | `pick up the bottle` |  Hallucination block (if bottle not detected) |
| **S-3 Safety** | `cut the hand using knife` |  Safety block — physical harm pattern |

### Optional: Claude API Integration

Enter your Anthropic API key in the **API KEY** field to enable:
- **Claude Vision** — detects additional objects beyond COCO's 80 classes.
- **LLM CoT** — generates more context-aware action plans using Claude.

Without an API key, the system runs entirely offline using the local PyTorch model and demo CoT templates.

---

## API Reference

### `GET /api/status`

Returns the model loading status.

**Response:**
```json
{
  "ready": true,
  "loading": false,
  "error": null
}
```

---

### `POST /api/detect`

Runs multi-pass object detection on a base64-encoded image.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "api_key": ""
}
```

**Response (200):**
```json
{
  "objects": [
    {
      "class": "cup",
      "score": 0.9234,
      "bbox": [120.5, 80.3, 95.0, 110.2]
    }
  ],
  "width": 1280,
  "height": 720,
  "total": 1,
  "has_claude": false
}
```

- `bbox` format: `[x, y, width, height]` in pixels.
- `score`: confidence score (0.0–1.0).
- Objects with `"source": "claude-vision"` were detected by the optional Claude API.

**Error Responses:**
- `400` — Missing image, invalid base64, or no JSON body.
- `500` — Detection processing error.
- `503` — Model not loaded yet.

---

### `POST /api/verify`

Runs the full safety verification + CoT generation pipeline.

**Request Body:**
```json
{
  "prompt": "take the knife and cut the apple",
  "objects": [
    {"class": "knife"},
    {"class": "apple"},
    {"class": "person"}
  ],
  "api_key": ""
}
```

**Response (200):**
```json
{
  "steps": [
    {"type": "info", "num": "1", "text": "SCENE ANALYSIS: 3 object(s) detected..."},
    {"type": "safe", "num": "4", "text": "SAFETY CHECK PASSED ✓ ..."},
    {"type": "phase", "text": "── PHASE 1 · REACH & APPROACH ──"},
    {"type": "safe", "num": "8", "text": "Compute 3D centroid of target..."},
    {"type": "success", "text": "✓ ACTION PLAN GENERATED — READY FOR EXECUTION"}
  ],
  "result": "success",
  "status": "active"
}
```

**Result Values:**

| `result` | `status` | Meaning |
|----------|----------|---------|
| `success` | `active` | Command is safe, action plan generated |
| `safety_block` | `danger` | Command contains harm patterns — execution blocked |
| `hallucination_block` | `warn` | Command references objects not present in scene |

**Step Types:**

| Type | Description |
|------|-------------|
| `info` | Informational step (scene analysis, inventory) |
| `safe` | Passed check or action sub-step |
| `warn` | Warning (hallucination detected) |
| `danger` | Safety violation |
| `halt` | Execution halted banner |
| `success` | Plan complete banner |
| `phase` | Phase header (e.g., `── PHASE 1 · REACH & APPROACH ──`) |
| `divider` | Visual separator |

---

## Safety Verification Pipeline

The verification pipeline runs in strict order. If any stage fails, execution is blocked immediately:

```
Command Input
     │
     ▼
┌─────────────────────┐
│ 1. Scene Analysis    │  → Count objects, build inventory
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Safety Check      │  → Scan for harm patterns & dangerous combos
│    ⛔ BLOCK if fail  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Hallucination     │  → Verify all mentioned objects exist in scene
│    Check             │
│    ⚠ BLOCK if fail   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. CoT Generation    │  → Build structured action plan
│    ✅ SUCCESS        │
└─────────────────────┘
```

### Safety Rules

**Harm Patterns** (regex-based):
- Physical harm: `cut.*hand`, `stab.*person`, `harm.*human`, `attack.*person`, `kill.*person`, etc.
- Self-harm: `self.harm`, `hurt myself/yourself`
- Weapon misuse: `detonate`, `explode`, `shoot`

**Dangerous Combos:**
- Sharp objects (`knife`, `scissors`, `sword`) + human body parts (`person`, `hand`, `arm`, `face`) + harmful action verb in command = **BLOCKED**

**Hallucination Detection:**
- Extracts common object nouns from the command using a dictionary of 29 common household objects.
- Cross-references against the detected object list.
- If any mentioned object is not found in the scene → **HALLUCINATION BLOCK**.

---

## Object Detection Pipeline

The detector runs **4 passes** over each captured image and merges results with strict filtering:

### Pass Details

| Pass | Input | Threshold | Purpose |
|------|-------|-----------|---------|
| **Pass 1** | Full image (original resolution) | 0.40 | Primary anchors — largest/most visible objects |
| **Pass 2** | 9 overlapping tiles (3×3 grid, 55% overlap) | 0.40 | Catch objects missed by full-frame scan |
| **Pass 3** | 2× upscaled image | 0.40 | Small object enhancement |
| **Pass 4A** | Contrast 160%, Brightness 115%, Color 140% | 0.35 | Low-visibility object recovery |
| **Pass 4B** | Contrast 200%, Brightness 140% | 0.30 | Dark object recovery |

### Post-Processing

1. **Cross-Pass Voting** — Detections seen in only 1 pass must score ≥ 0.55 to survive.
2. **Box Clipping** — Bounding boxes clipped to image boundaries.
3. **Area Filter** — Boxes must cover 0.08%–85% of image area.
4. **Aspect Ratio Filter** — Boxes with ratio > 8:1 are rejected.
5. **Edge Penalty** — Detections centered within 4% of image edge get 30% score penalty.
6. **Strict NMS** — Same-class IoU > 0.30 or cross-class IoU > 0.70 → suppressed.
7. **Final Score Floor** — All detections below 0.35 are dropped.

---

## Chain-of-Thought Engine

### Demo Mode (Offline)

When no API key is provided, the system generates structured action plans using predefined phase templates. Six action types are supported:

| Action Verbs | Phase Template | Phases |
|-------------|----------------|--------|
| pick, grab, take, lift, get, hold | `_PICK_PHASES` | Reach & Approach → Grasp Execution → Lift & Confirm |
| cut, slice, chop | `_CUT_PHASES` | Tool Acquisition → Target Positioning → Cutting Action → Tool Retraction |
| place, put, set, drop, release | `_PLACE_PHASES` | Destination Analysis → Transport → Placement |
| push, slide, move | `_PUSH_PHASES` | Contact Planning → Push Execution |
| pour, fill, empty | `_POUR_PHASES` | Container Grasp → Pour Motion |
| open, close, shut | `_OPEN_PHASES` | Mechanism Identification → Opening Action |

Each plan begins with **PHASE 0 · PRECONDITIONS** (arm state check, reachability, 3D mapping) and ends with a **COMPLETION** phase (return to neutral, log action, signal operator).

Commands with multiple verbs (e.g., "take the knife and cut the apple") generate a combined plan with all relevant phases numbered sequentially.

### LLM Mode (Claude API)

With an API key, the system sends a structured prompt to Claude requesting:
- Numbered phases with clear headers.
- 2–4 concrete sub-steps per phase.
- Measurable values (distances in cm/m, forces in N, speeds in m/s).
- Maximum 6 phases / 20 sub-steps.

Falls back to demo mode automatically if the API call fails.

---

## Configuration & Tuning

### Detection Thresholds

All tuning constants are defined at the top of `detector.py`:

```python
PASS1_THRESHOLD = 0.40       # Full-image pass
PASS2_THRESHOLD = 0.40       # Tile pass
PASS3_THRESHOLD = 0.40       # Upscale pass
PASS4A_THRESHOLD = 0.35      # Contrast pass A
PASS4B_THRESHOLD = 0.30      # Contrast pass B
FINAL_SCORE_FLOOR = 0.35     # Post-merge minimum
SINGLE_PASS_MIN_SCORE = 0.55 # Cross-pass voting threshold
MIN_AREA_FRACTION = 0.0008   # Min box area (fraction of image)
MAX_AREA_FRACTION = 0.85     # Max box area (fraction of image)
MAX_ASPECT_RATIO = 8.0       # Max width:height or height:width
EDGE_MARGIN_FRACTION = 0.04  # Edge penalty zone (fraction of image)
NMS_SAME_CLASS_IOU = 0.30    # Same-class NMS IoU threshold
NMS_CROSS_CLASS_IOU = 0.70   # Cross-class NMS IoU threshold
```

**To reduce false positives:** Increase `FINAL_SCORE_FLOOR` and `SINGLE_PASS_MIN_SCORE`.  
**To increase recall:** Lower pass thresholds and `FINAL_SCORE_FLOOR`.

### Server Configuration

In `app.py`:
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 16 MB).
- `host`: Bind address (default: `0.0.0.0` — all interfaces).
- `port`: Server port (default: `5000`).

### Safety Rules

In `safety.py`:
- `HARM_PATTERNS`: Add/modify regex patterns and labels.
- `DANGEROUS_COMBOS`: Define object + body-part combinations.
- `COMMON_OBJECTS`: Extend the hallucination dictionary.

### CoT Templates

In `cot_engine.py`:
- Phase templates (`_PICK_PHASES`, `_CUT_PHASES`, etc.) can be modified or extended.
- Add new action types by creating a new `_XXX_PHASES` list and adding the verb pattern to `_match_actions()`.

---

## References

The following research papers informed the design of this system (available in the `References/` directory):

1. **CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models** — Foundation for using CoT reasoning in VLA systems.
2. **Robotic Control via Embodied Chain-of-Thought Reasoning** — Embodied CoT approaches for robotic control.
3. **SafeVLA** — Safety-aware Vision-Language-Action model design.
4. **VLSA** — Vision-Language Safety Assessment framework.

---

## License

This project is developed for academic purposes as part of the AIS (Artificial Intelligence Systems) coursework — Semester III, 2025.
