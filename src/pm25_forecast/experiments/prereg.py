"""Pre-registration record for an experiment (KAN-79).

The schema mirrors ``docs/experiments/prereg_template.yaml``. Files are YAML or JSON. PyYAML is
only available as a transitive dev dependency (pre-commit), so it is imported lazily: JSON files
always load; YAML files need PyYAML in the environment.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_HORIZONS = (6, 12, 24)
MAX_FINALISTS = 2
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_TICKET_RE = re.compile(r"^KAN-\d+$")


class ExpectedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    direction: Literal["negative"]
    magnitude: float

    @field_validator("magnitude")
    @classmethod
    def _below_zero(cls, v: float) -> float:
        if not math.isfinite(v) or v >= 0:
            raise ValueError("magnitude must be a finite number below 0")
        return v


class Prereg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    ticket: str
    hypothesis: str = Field(min_length=10)
    model_family: str = Field(min_length=1)
    model: str = Field(min_length=1)
    feature_set: str = Field(min_length=1)
    params: dict[str, Any]
    horizons: list[int] = Field(min_length=1)
    expected_effect: ExpectedEffect
    decision_rule: str = Field(min_length=10)
    split: Literal["dev", "holdout"]
    final: bool
    finalists: int = Field(default=1, ge=1, le=MAX_FINALISTS)
    author_agent: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.fullmatch(v):
            raise ValueError("id must match ^[a-z0-9][a-z0-9_-]{2,63}$")
        return v

    @field_validator("ticket")
    @classmethod
    def _ticket(cls, v: str) -> str:
        if not _TICKET_RE.fullmatch(v):
            raise ValueError("ticket must look like KAN-<n>")
        return v

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v) or any(h not in ALLOWED_HORIZONS for h in v):
            raise ValueError(f"horizons must be unique values from {ALLOWED_HORIZONS}")
        if 24 not in v:
            raise ValueError("horizon 24 is the primary horizon and is mandatory")
        return sorted(v)

    @model_validator(mode="after")
    def _final_needs_holdout(self) -> "Prereg":
        if self.final and self.split != "holdout":
            raise ValueError("final=true requires split=holdout")
        return self


def parse_prereg(text: str, *, suffix: str = ".yaml") -> Prereg:
    """Parse and validate pre-registration text (``.json`` or YAML)."""
    if suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyYAML is not installed; use a .json pre-registration") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("pre-registration must be a mapping")
    return Prereg.model_validate(data)


def load_prereg(path: str | Path) -> Prereg:
    p = Path(path)
    return parse_prereg(p.read_text(encoding="utf-8"), suffix=p.suffix.lower())


def check_finalists(preregs: list[Prereg]) -> None:
    """Reject a set with duplicate ids or more than MAX_FINALISTS final candidates."""
    ids = [p.id for p in preregs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pre-registration ids")
    n_final = sum(p.final for p in preregs)
    if n_final > MAX_FINALISTS:
        raise ValueError(f"at most {MAX_FINALISTS} finalists may be registered, got {n_final}")
    wrong = [p.id for p in preregs if p.final and p.finalists != n_final]
    if wrong:
        raise ValueError(f"finalists must equal the number registered ({n_final}): {wrong}")
