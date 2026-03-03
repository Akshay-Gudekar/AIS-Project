"""
cot_engine.py - Chain-of-Thought generation engine.
Produces clear, structured, human-readable reasoning chains for robot actions.
  - Demo CoT generation (no API key) — detailed multi-phase action plans
  - Claude LLM CoT generation (with API key) — structured prompt for quality output
  - Claude Vision detection (for hybrid detection)
  - Deduplication of Claude vision results against COCO results
"""

import re
import json
import requests


# ─── ACTION VERB ↔ PHASE MAPPING ──────────────────────────────────────────────
# Maps action keywords to structured reasoning phases.
# Each phase: (header, list of steps)  — makes CoT self-explanatory.

_PICK_PHASES = [
    ("REACH & APPROACH", [
        "Compute 3D centroid of target object using stereo depth map.",
        "Plan collision-free trajectory from current end-effector pose to 5 cm above target.",
        "Execute approach at 0.3 m/s with proximity sensors active.",
    ]),
    ("GRASP EXECUTION", [
        "Analyse object geometry → select optimal grasp type (parallel-jaw / envelope).",
        "Open gripper to object width + 2 cm clearance.",
        "Lower onto grasp axis, close gripper to 12 N contact force.",
        "Verify stable hold via tactile feedback (no slip signal).",
    ]),
    ("LIFT & CONFIRM", [
        "Raise object 15 cm above surface at 0.1 m/s.",
        "Check for unexpected drag or tether — abort if payload > 20 N.",
        "Object securely held — ready for next task or placement.",
    ]),
]

_CUT_PHASES = [
    ("TOOL ACQUISITION", [
        "Locate knife in scene; confirm blade orientation pointing away from operator.",
        "Grasp knife handle with firm 18 N grip — wrist aligned to blade plane.",
    ]),
    ("TARGET POSITIONING", [
        "Verify target object is on a stable, flat surface.",
        "Scan cutting path for obstructions or fragile items — clear zone confirmed.",
    ]),
    ("CUTTING ACTION", [
        "Move blade tip to start-of-cut position 2 mm above target surface.",
        "Apply controlled downward force (8 N) along cutting axis at 0.02 m/s.",
        "Monitor force-torque sensor continuously; stop if resistance spike > 25 N.",
    ]),
    ("TOOL RETRACTION", [
        "Lift blade 10 cm clear of target surface.",
        "Rotate wrist to blade-safe orientation; return knife to original position.",
    ]),
]

_PLACE_PHASES = [
    ("DESTINATION ANALYSIS", [
        "Identify placement zone from command context (table / shelf / container).",
        "Check surface flatness and free area ≥ object footprint + 3 cm margin.",
    ]),
    ("TRANSPORT", [
        "Plan smooth path to destination at 0.2 m/s with vibration dampening.",
        "Maintain level orientation — tilt < 5° — to prevent contents from spilling.",
    ]),
    ("PLACEMENT", [
        "Descend toward surface at 0.05 m/s until contact detected by force sensor.",
        "Open gripper gradually (2 N residual) to release object without bounce.",
        "Retract gripper 10 cm vertically; confirm object remains stable for 1 s.",
    ]),
]

_PUSH_PHASES = [
    ("CONTACT PLANNING", [
        "Identify push-face on object; select flat end-effector surface.",
        "Approach push-face at 0.15 m/s until contact.",
    ]),
    ("PUSH EXECUTION", [
        "Apply lateral force (6 N) along push direction at 0.1 m/s.",
        "Monitor slip — adjust force if object rotates instead of translating.",
    ]),
]

_POUR_PHASES = [
    ("CONTAINER GRASP", [
        "Grasp container with stable cylindrical grip at centre of mass height.",
        "Verify lid / cap removed or open.",
    ]),
    ("POUR MOTION", [
        "Move container above target receptacle.",
        "Tilt wrist 90° over 2 s; monitor liquid flow via force-torque change.",
        "Return container upright once target fill level reached.",
    ]),
]

_OPEN_PHASES = [
    ("MECHANISM IDENTIFICATION", [
        "Identify opening mechanism (hinge / slide / twist / lift).",
        "Position end-effector at handle or grip point.",
    ]),
    ("OPENING ACTION", [
        "Apply appropriate motion: rotate (hinged), translate (sliding), or lift.",
        "Modulate force to avoid damage — max 15 N for household objects.",
    ]),
]


def _match_actions(prompt_lower):
    """Return list of (phase_header, steps) tuples for the prompt."""
    phases = []
    if re.search(r'pick|grab|take|lift|get|hold', prompt_lower):
        phases.extend(_PICK_PHASES)
    if re.search(r'cut|slice|chop', prompt_lower):
        phases.extend(_CUT_PHASES)
    if re.search(r'place|put|set|drop|release', prompt_lower):
        phases.extend(_PLACE_PHASES)
    if re.search(r'push|slide|move', prompt_lower):
        phases.extend(_PUSH_PHASES)
    if re.search(r'pour|fill|empty', prompt_lower):
        phases.extend(_POUR_PHASES)
    if re.search(r'open|close|shut', prompt_lower):
        phases.extend(_OPEN_PHASES)
    return phases


def generate_demo_steps(prompt, objects):
    """
    Generate a structured, easily understandable demo CoT plan.
    Organised into clear phases with descriptive step text.

    Args:
        prompt: The user's command string
        objects: List of object name strings

    Returns:
        List of step strings (already formatted for display)
    """
    p = prompt.lower()
    steps = []

    # ── Phase 0: Pre-conditions ─────────────────────────────────────────────
    target_obj = next((o for o in objects if o.lower() in p), objects[0] if objects else 'object')
    steps.append(f'── PHASE 0 · PRECONDITIONS ──')
    steps.append(f'Verify robot arm is in neutral home position (joints at 0°).')
    steps.append(f'Confirm target object [ {target_obj.upper()} ] is reachable within workspace radius (0.85 m).')
    steps.append(f'Build 3D spatial map of scene using depth camera + point-cloud fusion.')

    # ── Action-specific phases ──────────────────────────────────────────────
    action_phases = _match_actions(p)
    next_phase = 1  # Track phase numbering across sections

    if not action_phases:
        # Fallback for unrecognised verbs
        steps.append(f'── PHASE {next_phase} · ACTION PLANNING ──')
        next_phase += 1
        steps.append(f'Parse command: "{prompt}".')
        steps.append(f'No specific action template matched — generating generic safe approach.')
        steps.append(f'Navigate toward [ {target_obj.upper()} ] at cautious speed (0.15 m/s).')
        steps.append(f'Await further instruction or sensor confirmation before contact.')

    for idx, (header, phase_steps) in enumerate(action_phases):
        steps.append(f'── PHASE {next_phase} · {header} ──')
        next_phase += 1
        for s in phase_steps:
            # Substitute target object name where helpful
            steps.append(s)

    # ── Final phase: Completion ─────────────────────────────────────────────
    steps.append(f'── PHASE {next_phase} · COMPLETION ──')
    steps.append('Return end-effector to neutral standby posture.')
    steps.append('Log executed action sequence to episodic memory for future learning.')
    steps.append('Signal task completion to operator — ready for next command.')

    return steps


def llm_cot(prompt, objects, api_key):
    """
    Generate CoT using Claude API with a structured, readable prompt.

    Args:
        prompt: The user's command string
        objects: List of object name strings
        api_key: Anthropic API key

    Returns:
        dict with 'success' (bool), 'steps' (list) or 'error' (str), 'fallback' (bool)
    """
    system_prompt = """You are a VLA (Vision-Language-Action) robot safety reasoning engine.

Generate a Chain-of-Thought action plan that is **clear, structured, and easy for a human reviewer to understand**.

FORMAT RULES:
- Organise into numbered phases: "── PHASE N · PHASE_TITLE ──"
- Under each phase, write 2-4 concrete sub-steps.
- Use plain English — avoid jargon unless it adds precision.
- Include measurable values where relevant (distances in cm/m, forces in N, speeds in m/s).
- Start with precondition checks, then action phases, end with a completion phase.
- Maximum 6 phases / 20 sub-steps total.
- Output ONLY the plan — no preamble, no markdown, no commentary."""

    user_msg = f"""Scene objects (visually confirmed): {', '.join(objects)}
Operator command: "{prompt}"

Generate the structured robot action plan:"""

    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 800,
                'system': system_prompt,
                'messages': [{'role': 'user', 'content': user_msg}]
            },
            timeout=30
        )

        if resp.status_code != 200:
            error_data = {}
            try:
                error_data = resp.json()
            except Exception:
                pass
            error_msg = error_data.get('error', {}).get('message', resp.reason)
            return {'success': False, 'error': f'API Error: {error_msg}', 'fallback': True}

        data = resp.json()
        text = data.get('content', [{}])[0].get('text', '')
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        steps = []
        for line in lines:
            clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if clean:
                steps.append(clean)

        return {'success': True, 'steps': steps}

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out', 'fallback': True}
    except Exception as e:
        return {'success': False, 'error': f'Network error: {str(e)}', 'fallback': True}


def claude_vision_detect(image_base64, api_key, existing_preds, W, H):
    """
    Use Claude Vision to detect additional objects not found by COCO model.
    Matches the original JS claudeVisionDetect() function exactly.
    
    Args:
        image_base64: Base64-encoded image data (without data URL prefix)
        api_key: Anthropic API key
        existing_preds: List of already detected object dicts
        W: Image width in pixels
        H: Image height in pixels
    
    Returns:
        List of detected object dicts with 'source': 'claude-vision'
    """
    existing = ', '.join([p['class'] for p in existing_preds]) if existing_preds else 'none'

    system_prompt = f"""You are a meticulous computer vision labeler. 
Analyze the image carefully and list EVERY distinct visible object.
Already detected (skip these): {existing}.

Respond ONLY with a valid JSON array. No other text. No markdown.
Each object: {{
  "class": "object name in lowercase",
  "confidence": 0.0-1.0,
  "x_pct": center x as 0.0-1.0 fraction of image width,
  "y_pct": center y as 0.0-1.0 fraction of image height,
  "w_pct": width as 0.0-1.0 fraction of image width,
  "h_pct": height as 0.0-1.0 fraction of image height
}}

Be exhaustive. Specifically look for: bottles, mugs/cups, mobile phones/smartphones, bags/handbags, mirrors, tables, chairs, beds, windows, doors, lamps, shelves, cables/wires, clothing items, food items, remote controls, books, plants. 
Include partially visible objects. Minimum confidence: 0.45."""

    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 1200,
                'system': system_prompt,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': image_base64
                            }
                        },
                        {
                            'type': 'text',
                            'text': 'List every visible object as a JSON array with pixel-fraction coordinates.'
                        }
                    ]
                }]
            },
            timeout=30
        )

        if resp.status_code != 200:
            print(f"[cot_engine] Claude Vision API error: {resp.status_code}")
            return []

        data = resp.json()
        text = data.get('content', [{}])[0].get('text', '')

        # Parse JSON from response
        json_match = re.search(r'\[[\s\S]*?\]', text)
        if not json_match:
            return []

        items = json.loads(json_match.group())
        results = []
        for it in items:
            if not it or it.get('confidence', 0) < 0.45 or not it.get('class'):
                continue

            cx = it.get('x_pct', 0.5) * W
            cy = it.get('y_pct', 0.5) * H
            bw = it.get('w_pct', 0.2) * W
            bh = it.get('h_pct', 0.2) * H

            results.append({
                'class': str(it['class']).lower().strip(),
                'score': round(float(it.get('confidence', 0.5)), 4),
                'bbox': [
                    max(0, cx - bw / 2),
                    max(0, cy - bh / 2),
                    min(bw, W),
                    min(bh, H)
                ],
                'source': 'claude-vision'
            })

        return results

    except Exception as e:
        print(f"[cot_engine] Claude Vision detect failed: {e}")
        return []


def deduplicate_against_existing(coco_list, claude_list):
    """
    Remove Claude results that duplicate existing COCO detections.
    Matches the original JS deduplicateAgainstExisting() function exactly.
    
    Args:
        coco_list: List of COCO detection dicts
        claude_list: List of Claude Vision detection dicts
    
    Returns:
        Combined deduplicated list
    """
    coco_classes = set(p['class'].lower() for p in coco_list)
    aliases = {'mug': 'cup', 'cell phone': 'cell phone', 'monitor': 'tv'}

    filtered = []
    for c in claude_list:
        cl = c['class'].lower()
        # Drop if exact same class already found by COCO
        if cl in coco_classes:
            continue
        # Drop common mismatches
        if cl in aliases and aliases[cl] in coco_classes:
            continue
        filtered.append(c)

    return coco_list + filtered
