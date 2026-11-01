"""Fold and holdout date arithmetic of docs/experiments/protocol.md (section 4 and 7)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

S0 = datetime(2026, 7, 31, 17, tzinfo=UTC)  # first air4thai observation
TARGET_CUT = datetime(2026, 10, 26, tzinfo=UTC)  # KAN-74: targets from here on are excluded
FOLD = timedelta(days=7)
HOLDOUT = timedelta(days=21)
INITIAL_TRAIN = timedelta(days=28)
MIN_DEV = INITIAL_TRAIN + 3 * FOLD  # 49 days

MIN_HOLDOUT_PAIRS = 2100
MIN_HOLDOUT_ORIGINS = 400
MIN_DEV_FOLD_PAIRS = 600
MIN_STATION_PAIRS = 250


@dataclass(frozen=True)
class Folds:
    end: datetime  # E, exclusive end of the last full UTC day with targets
    holdout_start: datetime | None  # H0; None when the snapshot is too short
    holdout_folds: tuple[datetime, ...]  # weekly fold starts inside the holdout
    dev_folds: tuple[datetime, ...]  # weekly fold starts before H0 (oldest first)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("datetimes must be timezone-aware UTC")
    return value.astimezone(UTC)


def derive_folds(last_target: datetime, s0: datetime = S0) -> Folds:
    """Fold starts from the snapshot's last usable target time (protocol section 4.1)."""
    last_target, s0 = _utc(last_target), _utc(s0)
    limit = TARGET_CUT - timedelta(hours=1)
    end = (min(last_target, limit) + timedelta(hours=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    h0 = end - HOLDOUT
    if h0 - s0 < MIN_DEV:
        # Too short: no holdout; dev-only folds after the initial training window.
        starts, k = [], 0
        while end - FOLD * (k + 1) >= s0 + INITIAL_TRAIN:
            k += 1
            starts.append(end - FOLD * k)
        return Folds(end, None, (), tuple(reversed(starts)))
    dev, k = [], 0
    while h0 - FOLD * (k + 1) >= s0 + INITIAL_TRAIN:
        k += 1
        dev.append(h0 - FOLD * k)
    return Folds(end, h0, tuple(h0 + FOLD * i for i in range(3)), tuple(reversed(dev)))
