import json, re, time

DIRECTIONS_RE = re.compile(
    r"Directions\s+([EW0])\s*,\s*([NS0])\s*,\s*(UP|DOWN|0)",
    re.IGNORECASE
)

def _parse_stage(q1_text: str) -> str:
    t = (q1_text or "").lower()
    if "egg" in t: return "egg"
    if "larvae" in t or "larva" in t or "settlement" in t: return "larvae"
    return "larvae"  # default

def _parse_dirs(q_text: str):
    if not q_text: return None
    m = DIRECTIONS_RE.search(q_text)
    if not m: return None
    return (m.group(1).upper(), m.group(2).upper(), m.group(3).upper())

def _sign_ok(val: float, dir_token: str, eps: float) -> int:
    if dir_token in ('E','N','UP'):
        return 1 if val is not None and val >  eps else 0
    if dir_token in ('W','S','DOWN'):
        return 1 if val is not None and val < -eps else 0
    # '0'
    return 1 if val is not None and abs(val) <= eps else 0

def compute_logic_reward(item: dict, eps: float = 0.1):
    """Return (reward in [0,1], fails: list[str]) for one particle output item."""
    fails = []
    br = item.get("brief_rationale", {}) or {}
    q1 = br.get("q1","")
    q2 = br.get("q2","")
    q4 = br.get("q4","")

    dx = item.get("dx", 0.0)
    dy = item.get("dy", 0.0)
    dz = item.get("dz", 0.0)

    stage = _parse_stage(q1)

    # Hard gate: Egg => all moves must be ~0
    if stage == "egg":
        if (abs(dx) > 0.1) or (abs(dy) > 0.1) or (abs(dz) > 0.1):
            fails.append("egg_moved_nonzero")
            return 0.0, fails
        return 1.0, fails

    # Larvae/Settlement
    d2 = _parse_dirs(q2)
    d4 = _parse_dirs(q4)
    if d2 is None or d4 is None:
        fails.append("missing_or_unparseable_directions")
        return 0.0, fails

    sx = _sign_ok(dx, d2[0], 0.1)
    sy = _sign_ok(dy, d2[1], 0.1)
    sz = _sign_ok(dz, d2[2], 0.1)
    if not sx: fails.append("dx_sign_mismatch_q2")
    if not sy: fails.append("dy_sign_mismatch_q2")
    if not sz: fails.append("dz_sign_mismatch_q2")

    sc = 1 if d2 == d4 else 0
    if not sc: fails.append("q2_q4_direction_mismatch")

    reward = 0.3*sx + 0.3*sy + 0.3*sz + 0.1*sc
    return reward, fails

def parse_model_json_array(text: str):
    """Extract a top-level JSON array from model text."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    data = json.loads(t)
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array.")
    return data

