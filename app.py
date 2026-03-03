"""
app.py - Flask web server for VLA-4 CoT Safety Verifier.
Serves the frontend and provides REST API endpoints for:
  - Model status checking
  - Multi-pass object detection
  - Chain-of-Thought safety verification
"""

from flask import Flask, render_template, request, jsonify
from detector import ObjectDetector
from safety import safety_check, hallucination_check
from cot_engine import (
    generate_demo_steps, llm_cot,
    claude_vision_detect, deduplicate_against_existing
)

app = Flask(__name__)

# Set max content length to 16MB for large base64 images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global detector instance
detector = ObjectDetector()


# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Return current model loading status."""
    return jsonify({
        'ready': detector.ready,
        'loading': detector.loading,
        'error': detector.error
    })


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Run multi-pass object detection on a captured image.
    Expects JSON: { image: "data:image/jpeg;base64,...", api_key: "sk-..." (optional) }
    Returns JSON: { objects: [...], width: W, height: H, total: N }
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    image_data = data.get('image')
    api_key = data.get('api_key', '').strip()

    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    if not detector.ready:
        return jsonify({'error': 'Model not loaded yet. Please wait.'}), 503

    try:
        # Validate image data format
        try:
            if ',' in image_data:
                test_b64 = image_data.split(',')[1]
            else:
                test_b64 = image_data
            import base64 as _b64
            _b64.b64decode(test_b64[:100], validate=True)
        except Exception:
            return jsonify({'error': 'Invalid image data. Expected a valid base64-encoded image.'}), 400

        # Run multi-pass COCO detection (4 passes + strict NMS)
        merged, W, H = detector.detect_multipass(image_data)

        # Claude Vision detection if API key is provided
        claude_objects = []
        if api_key:
            base64_data = image_data.split(',')[1] if ',' in image_data else image_data
            claude_objects = claude_vision_detect(base64_data, api_key, merged, W, H)

        # Combine and deduplicate
        all_final = deduplicate_against_existing(merged, claude_objects)

        return jsonify({
            'objects': all_final,
            'width': W,
            'height': H,
            'total': len(all_final),
            'has_claude': bool(api_key and claude_objects)
        })

    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500


@app.route('/api/verify', methods=['POST'])
def api_verify():
    """
    Run Chain-of-Thought safety verification on a command.
    Expects JSON: { prompt: "...", objects: [...], api_key: "sk-..." (optional) }
    Returns JSON: { steps: [...], result: "success"|"safety_block"|"hallucination_block", status: "..." }
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    prompt = data.get('prompt', '').strip()
    objects = data.get('objects', [])
    api_key = data.get('api_key', '').strip()

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    if not objects:
        return jsonify({'error': 'No objects provided'}), 400

    # Validate object format — each must have a 'class' key
    try:
        obj_names = [o['class'] for o in objects]
    except (KeyError, TypeError):
        return jsonify({'error': 'Invalid objects format. Each object must have a "class" key.'}), 400

    # Build the CoT step-by-step response
    steps = []

    # ── Step 1 & 2: Scene analysis ──────────────────────────────────────────
    steps.append({
        'type': 'info', 'num': '1',
        'text': f'SCENE ANALYSIS: {len(objects)} object(s) detected in visual field.'
    })
    steps.append({
        'type': 'info', 'num': '2',
        'text': f'INVENTORY: [ {" · ".join(o.upper() for o in obj_names)} ]'
    })

    # ── Step 3 & 4: Safety check ────────────────────────────────────────────
    steps.append({'type': 'divider'})
    steps.append({
        'type': 'info', 'num': '3',
        'text': 'SAFETY VERIFIER → Parsing command for potential harm vectors...'
    })

    safety = safety_check(prompt, objects)

    if not safety['safe']:
        steps.append({
            'type': 'danger', 'num': '4',
            'text': f'⚠ SAFETY VIOLATION: {safety["reason"]}'
        })
        steps.append({
            'type': 'danger', 'num': '5',
            'text': 'ACTION BLOCKED. The requested operation involves potential physical harm to a human or unsafe weapon usage.'
        })
        steps.append({
            'type': 'halt',
            'text': '⛔ ACTION EXECUTION HALTED — SAFETY POLICY VIOLATION'
        })
        return jsonify({
            'steps': steps,
            'result': 'safety_block',
            'status': 'danger'
        })

    steps.append({
        'type': 'safe', 'num': '4',
        'text': 'SAFETY CHECK PASSED ✓ — No direct harm patterns detected.'
    })

    # ── Step 5 & 6: Hallucination check ─────────────────────────────────────
    steps.append({'type': 'divider'})
    steps.append({
        'type': 'info', 'num': '5',
        'text': 'GROUNDING VERIFIER → Cross-referencing command objects with scene inventory...'
    })

    hall = hallucination_check(prompt, objects)

    if hall['missing']:
        missing_str = ', '.join(o.upper() for o in hall['missing'])
        steps.append({
            'type': 'warn', 'num': '6',
            'text': f'HALLUCINATION DETECTED: Object(s) [ {missing_str} ] referenced in command but NOT present in scene.'
        })
        steps.append({
            'type': 'warn', 'num': '7',
            'text': 'GROUNDING FAILURE: Cannot execute action on objects not in visual field. This violates the Reality Grounding Principle in safe VLA policy.'
        })
        steps.append({
            'type': 'warn', 'num': '8',
            'text': 'DECISION: Reject action to prevent hallucination-driven robot behavior. Possible cause: stale world model or incorrect user assumption.'
        })
        steps.append({
            'type': 'halt',
            'text': f'⚠ ACTION REJECTED — OBJECTS NOT GROUNDED IN SCENE: [ {", ".join(o.upper() for o in hall["missing"])} ]'
        })
        return jsonify({
            'steps': steps,
            'result': 'hallucination_block',
            'status': 'warn'
        })

    steps.append({
        'type': 'safe', 'num': '6',
        'text': 'GROUNDING CHECK PASSED ✓ — All referenced objects confirmed in scene.'
    })

    # ── Step 7+: CoT generation ─────────────────────────────────────────────
    steps.append({'type': 'divider'})
    steps.append({
        'type': 'info', 'num': '7',
        'text': 'LLM REASONING ENGINE → Generating action plan...'
    })

    def _format_cot_steps(raw_steps, start_num=8):
        """Convert raw CoT step strings into typed step dicts.
        Lines starting with '──' are phase headers; others are action steps."""
        formatted = []
        num = start_num
        for s in raw_steps:
            if s.startswith('──') or s.startswith('--'):
                formatted.append({'type': 'phase', 'text': s})
            else:
                formatted.append({'type': 'safe', 'num': str(num), 'text': s})
                num += 1
        return formatted

    if api_key:
        result = llm_cot(prompt, obj_names, api_key)
        if result['success']:
            steps.extend(_format_cot_steps(result['steps']))
            steps.append({
                'type': 'success',
                'text': '✓ ACTION PLAN GENERATED — READY FOR EXECUTION'
            })
        else:
            # API failed - fallback to demo mode
            steps.append({
                'type': 'warn', 'num': '!',
                'text': f'{result["error"]}. Falling back to demo mode.'
            })
            demo_steps = generate_demo_steps(prompt, obj_names)
            steps.extend(_format_cot_steps(demo_steps))
            steps.append({
                'type': 'success',
                'text': '✓ ACTION PLAN GENERATED — READY FOR EXECUTION'
            })
    else:
        # Demo mode (no API key)
        demo_steps = generate_demo_steps(prompt, obj_names)
        steps.extend(_format_cot_steps(demo_steps))
        steps.append({
            'type': 'success',
            'text': '✓ ACTION PLAN GENERATED — READY FOR EXECUTION'
        })

    return jsonify({
        'steps': steps,
        'result': 'success',
        'status': 'active'
    })


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  VLA-4 :: CoT Safety Verifier — Python Backend")
    print("  Loading Faster R-CNN MobileNet V3 (COCO) model...")
    print("=" * 60)
    detector.load_model_async()
    app.run(debug=True, host='0.0.0.0', port=5000)
