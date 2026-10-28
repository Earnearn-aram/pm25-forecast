from pathlib import Path


def test_eda_workflow_is_manual_read_only_and_aggregate_only() -> None:
    workflow = Path(".github/workflows/eda.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in workflow
    assert "python -m pm25_forecast.eda" in workflow
    assert "actions/upload-artifact@cf430e" in workflow
    assert "eda-report/kan-13-metrics.json" in workflow
    assert "raw/" not in workflow


def test_eda_workflow_pins_third_party_actions() -> None:
    workflow = Path(".github/workflows/eda.yml").read_text()
    uses_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert uses_lines
    assert all("@" in line and "# v" in line for line in uses_lines)
