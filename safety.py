"""
safety.py - Safety verification and hallucination detection.
Exact replication of the safety rules from the original JS implementation.
"""

import re

# ─── HARM PATTERNS ─────────────────────────────────────────────────────────────
# Same patterns as the original JS HARM_PATTERNS array
HARM_PATTERNS = [
    {
        'pattern': r'cut.*hand|stab.*person|harm.*human|hurt.*person|attack.*person|cut.*finger|injur|kill.*person|kill.*human|kill.*people',
        'label': 'PHYSICAL HARM TO HUMAN'
    },
    {
        'pattern': r'self.harm|hurt (me|myself|yourself)',
        'label': 'SELF-HARM INTENT'
    },
    {
        'pattern': r'detonate|explode|shoot',
        'label': 'DANGEROUS WEAPON MISUSE'
    },
]

# ─── DANGEROUS OBJECT COMBOS ──────────────────────────────────────────────────
# Same as the original JS DANGEROUS_COMBOS array
DANGEROUS_COMBOS = [
    {
        'objects': ['knife', 'scissors', 'sword'],
        'bodyParts': ['person', 'hand', 'arm', 'face'],
        'label': 'SHARP OBJECT + HUMAN DETECTED'
    },
]

# ─── COMMON OBJECTS FOR HALLUCINATION CHECK ────────────────────────────────────
# Same list as the original JS commonObjects array
COMMON_OBJECTS = [
    'bottle', 'cup', 'apple', 'orange', 'knife', 'fork', 'spoon', 'book',
    'phone', 'laptop', 'chair', 'table', 'ball', 'pen', 'scissors',
    'glass', 'bowl', 'plate', 'banana', 'sandwich', 'mouse', 'keyboard',
    'remote', 'clock', 'vase', 'teddy bear', 'box', 'bag', 'key'
]


def safety_check(prompt, objects):
    """
    Check prompt against safety rules.
    Matches the original JS safetyCheck() function exactly.
    
    Args:
        prompt: The user's command string
        objects: List of detected object dicts with 'class' key
    
    Returns:
        dict with 'safe' (bool), and optionally 'reason' and 'type'
    """
    obj_names = [o['class'].lower() for o in objects]

    # Check prompt against harm patterns
    for rule in HARM_PATTERNS:
        if re.search(rule['pattern'], prompt, re.IGNORECASE):
            return {'safe': False, 'reason': rule['label'], 'type': 'harm_prompt'}

    # Check dangerous object + human combinations in scene
    for combo in DANGEROUS_COMBOS:
        has_weapon = any(
            any(w in o for o in obj_names) for w in combo['objects']
        )
        has_human = any(
            any(b in o for o in obj_names) for b in combo['bodyParts']
        )
        if has_weapon and has_human:
            # Extra check: prompt must reference weapon action toward human
            if (re.search(r'cut|stab|slice|harm|hit|attack|use.*knife|use.*scissor', prompt, re.IGNORECASE) and
                    re.search(r'hand|person|human|arm|body', prompt, re.IGNORECASE)):
                return {'safe': False, 'reason': combo['label'], 'type': 'dangerous_combo'}

    return {'safe': True}


def hallucination_check(prompt, objects):
    """
    Check if prompt references objects not present in the scene.
    Matches the original JS hallucinationCheck() function exactly.
    
    Args:
        prompt: The user's command string
        objects: List of detected object dicts with 'class' key
    
    Returns:
        dict with 'missing' (list) and 'mentioned' (list)
    """
    obj_names = [o['class'].lower() for o in objects]

    # Find common objects mentioned in the prompt
    mentioned = [
        obj for obj in COMMON_OBJECTS
        if re.search(r'\b' + re.escape(obj) + r'\b', prompt, re.IGNORECASE)
    ]

    # Find which mentioned objects are NOT in the detected scene
    missing = [
        obj for obj in mentioned
        if not any(obj in o for o in obj_names)
    ]

    return {'missing': missing, 'mentioned': mentioned}
