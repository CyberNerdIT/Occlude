"""Pass 2: the VLM modesty judge.

Replaces SegFormer clothing segmentation + InsightFace gender/age + the
hand-tuned RuleEngine with one model that looks at a person and reasons. The
v1 stack failed precisely where reasoning was needed: it flipped clean-shaven
young men to female, blurred a CGI character, over-blurred crowds shown from
behind, and mis-aged children — because three brittle heads each guessed
independently and the rules defaulted to blur on every uncertainty.

A VLM judges each tracklet from its clearest frames and returns a structured
verdict with a reason. The modesty policy lives in :data:`JUDGE_PROMPT`
(the same policy docs/03-rule-design.md spelled out); :mod:`occlude.pipeline.decide`
then aggregates across frames and applies the child / over-blur policy.

Default model is Qwen2.5-VL-7B-Instruct via transformers — open weights,
Apache-2.0, runs on a single A100/H100, and verified to exist as of this
writing. The model id is swappable, so a newer Qwen3-VL drops in by id once
its availability is confirmed.
"""
from __future__ import annotations

import json
import re

from occlude.pipeline.tracklets import (
    AGE_ADULT,
    AGE_CHILD,
    AGE_UNKNOWN,
    SEX_FEMALE,
    SEX_MALE,
    Verdict,
)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# The whole modesty policy, stated for the model. It judges ONE person (the
# image is a crop of a single detected person, with a little context). Output
# is strict JSON so parsing is deterministic — the model does the judgment,
# code does everything else (Rule 5).
JUDGE_PROMPT = """\
You are inspecting ONE person shown in the image for an Islamic-modesty video filter.
Decide whether this person should be blurred, following these rules exactly.

First determine:
- is_human: true only if this is a real human being. A cartoon, CGI/animated
  avatar, statue, mannequin, drawing, or abstract human-shaped figure is NOT a
  real human (is_human=false).
- sex: "male", "female", or "unknown". Judge from the whole figure (build,
  hair, dress), not the face alone. A clean-shaven young man is still male.
- age_bracket: "child" if the person clearly appears to be under 13, otherwise "adult".

Then apply the modesty rule:
- A MALE is blurred only if skin shows between the navel and the knee
  (shirtless, shorts, or bare thighs). Otherwise he is NOT blurred.
- A FEMALE is blurred if any of these is visible: uncovered hair, bare arms,
  bare legs, or an exposed neck/chest. A woman fully covered (e.g. hijab over
  all hair, long sleeves, long garment) is NOT blurred.
- A non-human figure is never blurred (blur=false).
- A child (under 13) is never blurred (blur=false).
- People shown only from behind or in a crowd with no exposed skin and no
  visible hair-vs-covering distinction are NOT blurred just for being present.

If you genuinely cannot tell whether a real adult is modest, prefer blur=true.

Respond with ONLY a JSON object, no other text:
{"is_human": bool, "sex": "male"|"female"|"unknown", "age_bracket": "child"|"adult", "blur": bool, "reason": "<short>"}
"""


class VLMJudge:
    """Wraps a vision-language model behind ``judge_crops(images) -> Verdicts``."""

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL,
        *,
        device: str | None = None,
        max_new_tokens: int = 192,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device if device else "auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._torch = torch

    def judge_crops(self, images: list) -> list[Verdict | None]:
        """Judge a batch of single-person crops (PIL images).

        Returns one Verdict per image, or None where the model output could
        not be parsed (decide.py treats an all-None tracklet as over-blur).
        """
        if not images:
            return []
        texts = []
        for _ in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": JUDGE_PROMPT},
                    ],
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = self.processor(
            text=texts, images=[[img] for img in images],
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
        ]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [parse_judgment(t) for t in decoded]


def parse_judgment(text: str) -> Verdict | None:
    """Parse one VLM JSON response into a Verdict, or None if unparseable.

    Tolerant of the model wrapping JSON in prose or markdown fences: it
    extracts the outermost ``{...}`` block. Returning None (rather than a
    fabricated default) keeps a single bad sample from polluting the
    cross-frame aggregation — decide.py decides what no-judgment means.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    return Verdict(
        blur=_as_bool(data.get("blur"), default=True),
        is_human=_as_bool(data.get("is_human"), default=True),
        sex=_map_sex(data.get("sex")),
        age_bracket=_map_age(data.get("age_bracket")),
        reason=str(data.get("reason", "")),
        confidence=1.0,
    )


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default


def _map_sex(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("male", "m", "man", "boy"):
        return SEX_MALE
    if v in ("female", "f", "woman", "girl"):
        return SEX_FEMALE
    return None


def _map_age(value: object) -> str:
    if not isinstance(value, str):
        return AGE_UNKNOWN
    v = value.strip().lower()
    if v in ("child", "kid", "minor"):
        return AGE_CHILD
    if v in ("adult",):
        return AGE_ADULT
    return AGE_UNKNOWN
