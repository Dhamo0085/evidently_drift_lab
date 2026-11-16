"""Unit tests for drift_monitor.py on synthetic data (no network)."""
import json
import math

import pandas as pd
import pytest

import drift_monitor as dm
from conftest import CAT, NUM, make_frame


# --------------------------------------------------------------------- detection

def test_no_drift_when_distributions_match(same_large):
    result, _ = dm.run_drift(*same_large, NUM, CAT)
    assert result.n_columns == 3
    assert result.n_drifted == 0
    assert result.drift_share == 0.0
    assert result.dataset_drift is False


def test_numeric_shift_is_detected_only_in_that_column(shifted_large):
    result, _ = dm.run_drift(*shifted_large, NUM, CAT)
    assert result.drifted_columns == ["x"]
    assert result.n_drifted == 1
    assert result.drift_share == pytest.approx(1 / 3)


def test_categorical_shift_is_detected():
    ref = make_frame(3000, seed=1)
    cur = make_frame(3000, colors=("red", "blue", "green"), color_p=[0.2, 0.2, 0.6], seed=2)
    result, _ = dm.run_drift(ref, cur, NUM, CAT)
    assert "color" in result.drifted_columns
    assert "y" not in result.drifted_columns


def test_dataset_drift_when_half_or_more_columns_drift():
    ref = make_frame(3000, seed=1)
    cur = make_frame(3000, x_mean=2.0, colors=("green",), seed=2)
    result, _ = dm.run_drift(ref, cur, NUM, CAT)
    assert result.n_drifted == 2
    assert result.dataset_drift is True


def test_large_data_uses_distance_methods(shifted_large):
    result, _ = dm.run_drift(*shifted_large, NUM, CAT)
    by_col = {c.column: c for c in result.columns}
    assert by_col["x"].method == "Wasserstein distance (normed)"
    assert by_col["color"].method == "Jensen-Shannon distance"
    assert all(c.threshold == 0.1 for c in result.columns)
    assert by_col["x"].score > 0.1 > by_col["y"].score


def test_small_data_uses_statistical_tests():
    result, _ = dm.run_drift(make_frame(300, seed=1), make_frame(300, x_mean=1.0, seed=2), NUM, CAT)
    by_col = {c.column: c for c in result.columns}
    assert by_col["x"].method == "K-S p_value"
    assert by_col["x"].threshold == 0.05
    assert by_col["x"].drifted and by_col["x"].score < 0.05  # p-value: small means drift


def test_column_types_and_row_counts(shifted_large):
    result, _ = dm.run_drift(*shifted_large, NUM, CAT)
    assert {c.column: c.column_type for c in result.columns} == {
        "x": "numerical", "y": "numerical", "color": "categorical"}
    assert (result.n_reference, result.n_current) == (3000, 3000)


def test_extra_columns_are_ignored(shifted_large):
    ref, cur = shifted_large
    ref, cur = ref.assign(id=range(len(ref))), cur.assign(id=range(len(cur)))
    result, _ = dm.run_drift(ref, cur, NUM, CAT)
    assert "id" not in {c.column for c in result.columns}


def test_default_schema_is_the_notebooks():
    assert dm.ADULT_NUMERICAL == ["education-num", "age", "capital-gain", "hours-per-week", "capital-loss", "fnlwgt"]
    assert len(dm.ADULT_CATEGORICAL) == 9 and "class" in dm.ADULT_CATEGORICAL


# -------------------------------------------------------------------- validation

@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r, c: (r.iloc[0:0], c), "reference data has no rows"),
        (lambda r, c: (r, c.drop(columns=["y"])), "current data is missing column(s): y"),
        (lambda r, c: (r.assign(x="text"), c), "'x' in the reference data is not numeric"),
        (lambda r, c: (r, c.assign(y=float("nan"))), "'y' in the current data is entirely empty"),
    ],
    ids=["empty", "missing-column", "non-numeric", "all-nan"],
)
def test_validation_errors(mutate, message):
    ref, cur = mutate(make_frame(50, seed=1), make_frame(50, seed=2))
    with pytest.raises(dm.InputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        dm.run_drift(ref, cur, NUM, CAT)


def test_validation_rejects_overlapping_or_empty_schema():
    ref, cur = make_frame(50), make_frame(50)
    with pytest.raises(dm.InputError, match="both numerical and categorical"):
        dm.validate_frames(ref, cur, ["x"], ["x"])
    with pytest.raises(dm.InputError, match="no columns"):
        dm.validate_frames(ref, cur, [], [])


def test_infer_schema():
    num, cat = dm.infer_schema(make_frame(10))
    assert (num, cat) == (["x", "y"], ["color"])


# --------------------------------------------------------------------- scenarios

def test_education_scenario_matches_notebook_split(adult_like):
    ref, cur = dm.make_scenario(adult_like, "education")
    assert set(cur["education"]) <= set(dm.PRODUCTION_EDUCATION)
    assert not set(ref["education"]) & set(dm.PRODUCTION_EDUCATION)
    assert len(ref) + len(cur) == len(adult_like)


def test_random_scenario_is_two_disjoint_halves(adult_like):
    ref, cur = dm.make_scenario(adult_like, "random", seed=1)
    assert abs(len(ref) - len(cur)) <= 1
    assert not set(ref.index) & set(cur.index)


def test_age_shift_adds_ten_years_to_current_only(adult_like):
    ref, cur = dm.make_scenario(adult_like, "age_shift", seed=1)
    assert (cur["age"] - adult_like.loc[cur.index, "age"] == 10).all()
    assert (ref["age"] == adult_like.loc[ref.index, "age"]).all()


def test_unknown_scenario(adult_like):
    with pytest.raises(dm.InputError, match="unknown scenario"):
        dm.make_scenario(adult_like, "nope")


# ----------------------------------------------------------------------- outputs

def test_save_outputs(shifted_large, tmp_path):
    result, snapshot = dm.run_drift(*shifted_large, NUM, CAT, with_summary=True)
    paths = dm.save_outputs(result, snapshot, tmp_path / "out", "unit test")
    assert paths["html"].stat().st_size > 10_000
    summary = json.loads(paths["summary"].read_text())
    assert summary["label"] == "unit test"
    assert summary["drifted_columns"] == ["x"]
    assert summary["n_columns"] == 3
    columns = pd.read_csv(paths["columns"])
    assert list(columns.columns) == ["column", "column_type", "score", "method", "threshold", "drifted"]
    assert columns.set_index("column").loc["x", "drifted"]


def test_format_table_lists_drifted_first(shifted_large):
    result, _ = dm.run_drift(*shifted_large, NUM, CAT)
    lines = dm.format_table(result).splitlines()
    assert lines[1].startswith("x ") and "YES" in lines[1]


# ------------------------------------------------------------- batch monitoring

def test_gradual_batches_mix_increasingly_more_current():
    ref, cur = make_frame(2000, seed=1), make_frame(2000, x_mean=2.0, seed=2)
    baseline, batches = dm.gradual_drift_batches(ref, cur, n_batches=5, batch_size=400, seed=0)
    assert len(baseline) == 1000
    assert [round(f, 2) for f, _ in batches] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(len(b) == 400 for _, b in batches)
    means = [b["x"].mean() for _, b in batches]
    assert means == sorted(means)  # more shifted rows -> higher mean


def test_monitor_batches_drift_grows_and_chart_is_written(tmp_path):
    ref, cur = make_frame(4000, seed=1), make_frame(4000, x_mean=2.0, seed=2)
    baseline, batches = dm.gradual_drift_batches(ref, cur, n_batches=3, batch_size=1500, seed=0)
    trend = dm.monitor_batches(baseline, batches, NUM, CAT)
    assert list(trend["batch"]) == [0, 1, 2]
    assert trend["drift_share"].iloc[0] == 0.0
    assert trend["drift_share"].iloc[-1] > 0.0
    assert "x" in trend["drifted_columns"].iloc[-1]
    png = dm.plot_trend(trend, tmp_path / "trend.png", gate=0.3)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_batches_need_at_least_two():
    with pytest.raises(dm.InputError, match="at least 2"):
        dm.gradual_drift_batches(make_frame(10), make_frame(10), n_batches=1)


# ----------------------------------------------------------------------------- CLI

@pytest.fixture
def csv_pair(tmp_path, shifted_large):
    ref, cur = shifted_large
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    ref.to_csv(ref_path, index=False)
    cur.to_csv(cur_path, index=False)
    return ref_path, cur_path


def test_cli_csv_input_writes_reports(csv_pair, tmp_path, capsys):
    out = tmp_path / "reports"
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]), "--out", str(out)])
    assert code == 0
    assert {p.name for p in (out / "custom").iterdir()} == {"drift_report.html", "summary.json", "columns.csv"}
    assert "Drifted columns: 1/3" in capsys.readouterr().out


def test_cli_gate_breached_returns_1(csv_pair, tmp_path, capsys):
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]),
                    "--out", str(tmp_path), "--fail-above", "0.2"])
    assert code == 1
    assert "DRIFT ALERT" in capsys.readouterr().err


def test_cli_gate_not_breached_returns_0(csv_pair, tmp_path):
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]),
                    "--out", str(tmp_path), "--fail-above", "0.5"])
    assert code == 0


def test_cli_explicit_schema(csv_pair, tmp_path, capsys):
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]), "--out", str(tmp_path),
                    "--num-cols", "y", "--cat-cols", "color"])
    assert code == 0
    assert "Drifted columns: 0/2" in capsys.readouterr().out  # x (the shifted one) not monitored


def test_cli_batches_with_csv(csv_pair, tmp_path):
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]), "--out", str(tmp_path),
                    "--batches", "3", "--batch-size", "1000"])
    assert code == 0
    trend = pd.read_csv(tmp_path / "custom" / "drift_trend.csv")
    assert len(trend) == 3
    assert (tmp_path / "custom" / "drift_trend.png").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--fail-above", "1.5"], "--fail-above must be between 0 and 1"),
        (["--num-cols", "missing_col"], "missing column(s): missing_col"),
        (["--batches", "1"], "at least 2"),
    ],
    ids=["gate-out-of-range", "unknown-column", "too-few-batches"],
)
def test_cli_input_errors_return_2(csv_pair, tmp_path, capsys, extra, message):
    code = dm.main(["--reference", str(csv_pair[0]), "--current", str(csv_pair[1]),
                    "--out", str(tmp_path)] + extra)
    assert code == 2
    assert message in capsys.readouterr().err


def test_cli_reference_without_current(csv_pair, capsys):
    assert dm.main(["--reference", str(csv_pair[0])]) == 2
    assert "must be given together" in capsys.readouterr().err


def test_cli_missing_file(tmp_path, capsys):
    assert dm.main(["--reference", str(tmp_path / "nope.csv"), "--current", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_unknown_scenario():
    with pytest.raises(SystemExit) as exc:
        dm.main(["--scenario", "weekly"])
    assert exc.value.code == 2


def test_drift_share_is_finite_fraction(shifted_large):
    result, _ = dm.run_drift(*shifted_large, NUM, CAT)
    assert 0.0 <= result.drift_share <= 1.0 and not math.isnan(result.drift_share)
