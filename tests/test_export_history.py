from datetime import UTC, datetime

import duckdb

from pm25_forecast.export_history import export_history


class ReadOnlySource:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement):
        self.statements.append(statement)
        return self

    def fetchall(self):
        if "FROM observations" in self.statements[-1]:
            return [
                (
                    "59t",
                    datetime(2026, 1, 1, tzinfo=UTC),
                    12.5,
                    "air4thai",
                    datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                )
            ]
        return [
            (
                "59t",
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 6, tzinfo=UTC),
                "cams_global",
                "run-1",
                13.0,
                datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
            )
        ]


def test_export_is_read_only_and_preserves_natural_keys(tmp_path):
    source = ReadOnlySource()
    counts = export_history(tmp_path / "history.duckdb", "unused", connect=lambda _: source)

    assert source.statements[0] == "SET TRANSACTION READ ONLY"
    assert counts == {"observations": 1, "forecast_vintages": 1}
    with duckdb.connect(str(tmp_path / "history.duckdb"), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        observation = con.execute(
            "SELECT station_id, ts, pm25, source FROM observations"
        ).fetchone()
        vintage = con.execute(
            "SELECT station_id, issue_time, valid_time, model_version, run_id "
            "FROM forecast_vintages"
        ).fetchone()
    assert observation[:1] == ("59t",)
    assert observation[1].utcoffset().total_seconds() == 0
    assert observation[2] == 12.5
    assert observation[3] == "air4thai"
    assert vintage == (
        "59t",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        "cams_global",
        "run-1",
    )
