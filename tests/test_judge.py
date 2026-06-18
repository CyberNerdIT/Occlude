"""VLM judgment parsing — robust to messy model output, fails safe."""
from occlude.pipeline.judge import parse_judgment
from occlude.pipeline.tracklets import AGE_CHILD, SEX_FEMALE, SEX_MALE


def test_clean_json():
    v = parse_judgment(
        '{"is_human": true, "sex": "male", "age_bracket": "adult", '
        '"blur": false, "reason": "modest"}'
    )
    assert v is not None
    assert v.is_human is True
    assert v.sex == SEX_MALE
    assert v.blur is False


def test_json_wrapped_in_prose_and_fences():
    text = (
        "Here is my judgment:\n```json\n"
        '{"is_human": true, "sex": "female", "age_bracket": "adult", "blur": true, "reason": "uncovered hair"}'
        "\n```\nDone."
    )
    v = parse_judgment(text)
    assert v is not None
    assert v.sex == SEX_FEMALE
    assert v.blur is True


def test_string_booleans_accepted():
    v = parse_judgment('{"is_human": "true", "sex": "f", "age_bracket": "child", "blur": "false"}')
    assert v is not None
    assert v.is_human is True
    assert v.age_bracket == AGE_CHILD


def test_unknown_sex_maps_to_none():
    v = parse_judgment('{"sex": "unsure", "blur": true}')
    assert v is not None
    assert v.sex is None


def test_missing_blur_defaults_true():
    # Over-blur bias survives a model that omits the field.
    v = parse_judgment('{"is_human": true, "sex": "female"}')
    assert v is not None
    assert v.blur is True


def test_unparseable_returns_none():
    assert parse_judgment("no json here") is None
    assert parse_judgment("") is None
    assert parse_judgment("{ broken json,,,") is None
