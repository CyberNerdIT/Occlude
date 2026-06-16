"""Unit tests for RuleEngine — no models required.

Mask geometry (all tests unless noted):
  H=100, W=80  →  8 000 pixels total
  Face rows 20–40, cols 20–60  →  face_h=21, face_w=41, face_area=861
  Head region (derived): rows 0–40, cols 0–80  →  head_area=3 200
  Thigh region (derived): rows 92–100, cols 0–80

Threshold reference:
  HAIR_MIN_CC_FRAC  0.03  →  need CC ≥ 96 px in head region
  HATSCARF_HAIR_RATIO  0.5  →  blur when hatscarf ≤ 0.5 × hair
  HAIR_BELOW_FACE_PCT  1.0  →  blur when below-face hair ≥ 80 px (1% of 8 000)
  CROSSCHECK_HAIR_AREA_FACTOR  0.20  →  M→F when below-face hair > 172 px (0.2 × 861)
  SHIRTLESS_EPS_PCT  0.5  →  blur M when Upper-clothes < 40 px (0.5% of 8 000)
  ARMS_MIN_PCT  4.0  →  blur F when arm pixels > 320
  LEGS_MIN_PCT  2.0  →  blur F when leg pixels > 160
"""
import numpy as np
from PIL import Image

from occlude.pipeline.perception import SEG_LABELS, Perception, Person
from occlude.pipeline.rules import RuleEngine

L = {name: i for i, name in enumerate(SEG_LABELS)}

H, W = 100, 80
FACE_ROW_TOP, FACE_ROW_BOT = 20, 40
FACE_COL_LEFT, FACE_COL_RIGHT = 20, 60

_engine = RuleEngine()


def _person(
    seg_mask: np.ndarray,
    gender: str | None = "F",
    face_score: float = 0.8,
    age: float | None = None,
) -> Person:
    return Person(
        bbox=(0, 0, W, H),
        det_conf=0.9,
        crop=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
        seg_mask=seg_mask,
        gender=gender,
        face_det_score=face_score,
        label_masks=Perception.make_label_masks(seg_mask),
        age=age,
    )


def _base() -> np.ndarray:
    """Background-only mask with a Face block at the standard position."""
    m = np.zeros((H, W), dtype=np.int32)
    m[FACE_ROW_TOP:FACE_ROW_BOT + 1, FACE_COL_LEFT:FACE_COL_RIGHT + 1] = L["Face"]
    return m


# ---------------------------------------------------------------------------
# Gender override / default
# ---------------------------------------------------------------------------

def test_gender_none_defaults_to_female():
    d = _engine.decide(_person(_base(), gender=None, face_score=0.0))
    assert d.gender_used == "F"
    assert d.overridden is True


def test_low_face_score_defaults_to_female():
    # score 0.3 < MIN_FACE_DET_SCORE=0.5 → override
    d = _engine.decide(_person(_base(), gender="M", face_score=0.3))
    assert d.gender_used == "F"
    assert d.overridden is True


def test_crosscheck_fires_on_hair_below_face():
    # 18×41=738 px Hair below face_bot=40; threshold=172 px → override M→F
    m = _base()
    m[41:59, FACE_COL_LEFT:FACE_COL_RIGHT + 1] = L["Hair"]
    d = _engine.decide(_person(m, gender="M", face_score=0.8))
    assert d.gender_used == "F"
    assert d.overridden is True


def test_crosscheck_does_not_fire_on_sparse_hair_below_face():
    # 2×41=82 px Hair below face (< 172 threshold) — gender stays M
    m = _base()
    m[41:61, :] = L["Upper-clothes"]          # torso coverage (written first)
    m[41:43, FACE_COL_LEFT:FACE_COL_RIGHT + 1] = L["Hair"]  # 82 px
    d = _engine.decide(_person(m, gender="M", face_score=0.8))
    assert d.gender_used == "M"
    assert d.overridden is False


# ---------------------------------------------------------------------------
# No face anchor
# ---------------------------------------------------------------------------

def test_no_face_label_over_blurs():
    m = np.zeros((H, W), dtype=np.int32)   # no Face label anywhere
    d = _engine.decide(_person(m))
    assert d.blur is True
    assert "no Face" in d.reason


# ---------------------------------------------------------------------------
# Male rules
# ---------------------------------------------------------------------------

def test_male_shirtless_blurs():
    # 5 px Upper-clothes → 0.0625% < SHIRTLESS_EPS_PCT=0.5%
    m = _base()
    m[50, 30:35] = L["Upper-clothes"]
    d = _engine.decide(_person(m, gender="M", face_score=0.8))
    assert d.blur is True
    assert "shirtless" in d.reason


def test_male_covered_passes():
    # 39×60=2 340 px Upper-clothes (29%); thigh region (rows 92–99) is bare background
    m = _base()
    m[41:80, 10:70] = L["Upper-clothes"]
    d = _engine.decide(_person(m, gender="M", face_score=0.8))
    assert d.blur is False


# ---------------------------------------------------------------------------
# Female hair — head-region rules
# ---------------------------------------------------------------------------

def test_female_uncovered_hair_blurs():
    # 10×10=100 px Hair; CC frac=3.1% ≥ 3%; no Hat/Scarf → fires
    m = _base()
    m[5:15, 30:40] = L["Hair"]
    d = _engine.decide(_person(m))
    assert d.blur is True
    assert "uncovered hair" in d.reason


def test_female_covered_hair_passes_via_hatscarf():
    # Hair: 100 px; Hat: 300 px (no overlap) → hatscarf=300 > 0.5×100=50 → covered
    m = _base()
    m[5:15, 30:40] = L["Hair"]    # 10×10=100 px
    m[0:5, 20:80] = L["Hat"]      # 5×60=300 px, no row overlap with Hair
    d = _engine.decide(_person(m))
    assert d.blur is False


def test_female_small_hair_cc_does_not_trigger():
    # 5×5=25 px Hair → CC frac=0.78% < 3% → head-region rule does not fire
    m = _base()
    m[5:10, 30:35] = L["Hair"]
    d = _engine.decide(_person(m))
    assert d.blur is False


# ---------------------------------------------------------------------------
# Female hair — below-face rule
# ---------------------------------------------------------------------------

def test_female_hair_below_face_blurs():
    # 9×10=90 px Hair below face_bot=40 → 1.125% ≥ HAIR_BELOW_FACE_PCT=1.0%
    m = _base()
    m[45:54, 30:40] = L["Hair"]
    d = _engine.decide(_person(m))
    assert d.blur is True
    assert "hair below face" in d.reason


def test_female_hair_below_face_below_threshold_passes():
    # 3×10=30 px below face → 0.375% < 1.0%; no head-region hair → passes
    m = _base()
    m[45:48, 30:40] = L["Hair"]
    d = _engine.decide(_person(m))
    assert d.blur is False


# ---------------------------------------------------------------------------
# Female body rules
# ---------------------------------------------------------------------------

def test_female_bare_arms_blurs():
    # Left-arm: 17×20=340 px → 4.25% > ARMS_MIN_PCT=4.0%
    m = _base()
    m[50:67, 0:20] = L["Left-arm"]
    d = _engine.decide(_person(m))
    assert d.blur is True
    assert "bare arms" in d.reason


def test_female_bare_arms_below_threshold_passes():
    # Left-arm: 11×20=220 px → 2.75% < 4.0%
    m = _base()
    m[50:61, 0:20] = L["Left-arm"]
    d = _engine.decide(_person(m))
    assert d.blur is False


def test_female_bare_legs_blurs():
    # Left-leg: 17×30=510 px → 6.375% > LEGS_MIN_PCT=2.0%
    m = _base()
    m[60:77, 20:50] = L["Left-leg"]
    d = _engine.decide(_person(m))
    assert d.blur is True
    assert "bare legs" in d.reason


# ---------------------------------------------------------------------------
# Child exemption
# ---------------------------------------------------------------------------

def test_child_age_exempts_even_when_rules_would_blur():
    # Bare legs would normally blur a female; a child age must override.
    m = _base()
    m[60:77, 20:50] = L["Left-leg"]
    d = _engine.decide(_person(m, age=8.0))
    assert d.blur is False
    assert "child" in d.reason


def test_child_age_exempts_even_with_no_face_overblur():
    # No Face label → the over-blur bias normally forces blur=True.
    # The child gate must fire ahead of it.
    m = np.zeros((H, W), dtype=np.int32)
    d = _engine.decide(_person(m, gender=None, face_score=0.0, age=10.0))
    assert d.blur is False
    assert "child" in d.reason


def test_age_at_cutoff_is_exempt_but_above_is_not():
    m = _base()
    m[60:77, 20:50] = L["Left-leg"]
    assert _engine.decide(_person(m, age=12.0)).blur is False   # ≤ 12 exempt
    assert _engine.decide(_person(m, age=13.0)).blur is True     # adult band


def test_age_none_unchanged_behaviour():
    # No age signal → rules behave exactly as before (control).
    m = _base()
    m[60:77, 20:50] = L["Left-leg"]
    assert _engine.decide(_person(m, age=None)).blur is True
