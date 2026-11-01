"""KAN-79: pre-registration model and template."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pm25_forecast.experiments.prereg import (
    Prereg,
    check_finalists,
    load_prereg,
    parse_prereg,
)

TEMPLATE = Path(__file__).parent.parent / "docs" / "experiments" / "prereg_template.yaml"


def _good() -> dict:
    return load_prereg(TEMPLATE).model_dump()


def _make(**over) -> Prereg:
    d = _good()
    d.update(over)
    return Prereg.model_validate(d)


def test_template_is_valid():
    p = load_prereg(TEMPLATE)
    assert p.horizons == [6, 12, 24]
    assert p.final is False
    assert p.expected_effect.magnitude < 0


def test_json_roundtrip(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps(_good()))
    assert load_prereg(f).id == load_prereg(TEMPLATE).id


@pytest.mark.parametrize(
    "over",
    [
        {"id": "Bad Id"},
        {"id": "ab"},
        {"ticket": "ABC-1"},
        {"horizons": [6, 12]},
        {"horizons": [24, 24]},
        {"horizons": [24, 48]},
        {"horizons": []},
        {"split": "test"},
        {"final": True, "split": "dev"},
        {"hypothesis": "short"},
        {"extra_field": 1},
        {"expected_effect": {"metric": "m", "direction": "negative", "magnitude": 0.1}},
        {"expected_effect": {"metric": "m", "direction": "positive", "magnitude": -0.1}},
    ],
)
def test_rejects_bad(over):
    with pytest.raises(ValidationError):
        _make(**over)


def test_missing_field_rejected():
    d = _good()
    del d["decision_rule"]
    with pytest.raises(ValidationError):
        Prereg.model_validate(d)


def test_non_mapping_rejected():
    with pytest.raises(ValueError, match="mapping"):
        parse_prereg("- a\n- b\n")


def test_finalist_cap_and_duplicates():
    fin = [_make(id=f"cand-{i}", split="holdout", final=True) for i in range(3)]
    check_finalists(fin)
    with pytest.raises(ValueError, match="at most 3"):
        check_finalists([*fin, _make(id="cand-9", split="holdout", final=True)])
    with pytest.raises(ValueError, match="duplicate"):
        check_finalists([fin[0], fin[0]])
