"""
Data-drift monitor built on the lab's Evidently report.

Compares a *reference* dataset with a *current* one, saves the Evidently HTML
report plus machine-readable results, and can act as a CI/production gate.

Usage (from this folder):
    python drift_monitor.py                         # the notebook's scenario (education split)
    python drift_monitor.py --scenario random       # control: same distribution, expect no drift
    python drift_monitor.py --scenario age_shift    # synthetic drift: current population 10 years older
    python drift_monitor.py --reference ref.csv --current cur.csv
    python drift_monitor.py --fail-above 0.3        # exit code 1 if > 30% of columns drift
    python drift_monitor.py --batches 6             # gradual-drift stream, writes a trend chart

Exit codes: 0 = OK, 1 = drift gate breached, 2 = bad input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = LAB_DIR / "data_cache"
DEFAULT_OUT = LAB_DIR / "reports"

# Same schema as the notebook (cell 6).
ADULT_NUMERICAL = ["education-num", "age", "capital-gain", "hours-per-week", "capital-loss", "fnlwgt"]
ADULT_CATEGORICAL = ["education", "occupation", "native-country", "workclass", "marital-status",
                     "relationship", "race", "sex", "class"]
# The notebook's split: these three education levels form the "production" data.
PRODUCTION_EDUCATION = ["Some-college", "HS-grad", "Bachelors"]
SCENARIOS = ("education", "random", "age_shift")


class InputError(ValueError):
    """The data cannot be monitored (missing columns, empty, wrong types...)."""


@dataclass
class ColumnDrift:
    column: str
    column_type: str          # "numerical" or "categorical"
    score: float
    method: str               # chosen by Evidently, e.g. "Wasserstein distance (normed)"
    threshold: float
    drifted: bool


@dataclass
class DriftResult:
    n_reference: int
    n_current: int
    n_columns: int
    n_drifted: int
    drift_share: float        # share of columns that drifted
    dataset_drift: bool       # Evidently's verdict: drift_share >= the preset's drift_share (0.5)
    columns: list[ColumnDrift] = field(default_factory=list)

    @property
    def drifted_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.drifted]

    def to_dict(self) -> dict:
        return {**asdict(self), "drifted_columns": self.drifted_columns}


# ----------------------------------------------------------------------------- data

def load_adult(cache_dir: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """The notebook's dataset (OpenML 'adult' v2), cached locally after the first download."""
    from sklearn import datasets

    return datasets.fetch_openml(name="adult", version=2, as_frame="auto", data_home=str(cache_dir)).frame


def make_scenario(data: pd.DataFrame, scenario: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (reference, current) from one dataset.

    education : the notebook's split (current = 3 education levels) -> real drift
    random    : random 50/50 split -> same distribution, should show no drift (control)
    age_shift : random 50/50 split, then current ages +10 years -> drift in "age" only
    """
    if scenario == "education":
        in_prod = data["education"].isin(PRODUCTION_EDUCATION)
        return data[~in_prod], data[in_prod]
    if scenario not in SCENARIOS:
        raise InputError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")

    shuffled = data.sample(frac=1.0, random_state=seed)
    half = len(shuffled) // 2
    reference, current = shuffled.iloc[:half], shuffled.iloc[half:].copy()
    if scenario == "age_shift":
        current["age"] = current["age"] + 10
    return reference, current


def infer_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric dtypes -> numerical columns, everything else -> categorical."""
    numerical = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numerical, [c for c in df.columns if c not in numerical]


def validate_frames(reference: pd.DataFrame, current: pd.DataFrame,
                    numerical: list[str], categorical: list[str]) -> None:
    """Fail early with a readable message instead of an Evidently traceback."""
    if not numerical and not categorical:
        raise InputError("no columns to monitor")
    overlap = set(numerical) & set(categorical)
    if overlap:
        raise InputError(f"column(s) listed as both numerical and categorical: {sorted(overlap)}")
    for name, df in (("reference", reference), ("current", current)):
        if df.empty:
            raise InputError(f"the {name} data has no rows")
        missing = [c for c in numerical + categorical if c not in df.columns]
        if missing:
            raise InputError(f"the {name} data is missing column(s): {', '.join(missing)}")
        for col in numerical:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise InputError(f"column {col!r} in the {name} data is not numeric")
            if df[col].isna().all():
                raise InputError(f"column {col!r} in the {name} data is entirely empty")


# ------------------------------------------------------------------------ analysis

_DESCRIPTION = re.compile(r"The drift detection method is (?P<method>.+?)\. The drift threshold is (?P<threshold>\d+(?:\.\d+)?)")


def build_report(numerical: list[str], categorical: list[str], with_summary: bool) -> tuple[Report, DataDefinition]:
    presets = [DataDriftPreset()] + ([DataSummaryPreset()] if with_summary else [])
    return Report(presets, include_tests=True), DataDefinition(numerical_columns=numerical,
                                                               categorical_columns=categorical)


def run_drift(reference: pd.DataFrame, current: pd.DataFrame,
              numerical: list[str] | None = None, categorical: list[str] | None = None,
              with_summary: bool = False):
    """
    Run Evidently's DataDriftPreset (as in the notebook) and return (DriftResult, snapshot).

    Evidently picks the statistical method per column from the data size and
    type; the per-column drift decision is read from its own tests, so this
    never re-implements Evidently's rules.
    """
    if numerical is None and categorical is None:
        numerical, categorical = ADULT_NUMERICAL, ADULT_CATEGORICAL
    numerical, categorical = list(numerical or []), list(categorical or [])
    validate_frames(reference, current, numerical, categorical)

    report, definition = build_report(numerical, categorical, with_summary)
    cols = numerical + categorical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = report.run(Dataset.from_pandas(pd.DataFrame(current[cols]), data_definition=definition),
                              Dataset.from_pandas(pd.DataFrame(reference[cols]), data_definition=definition))

    payload = snapshot.dict()
    scores = {m["metric_id"]: m["value"] for m in payload["metrics"]}
    columns = []
    for test in payload["tests"]:
        params = test["metric_config"]["params"]
        if not params["type"].endswith("ValueDrift"):
            continue
        name = params["column"]
        match = _DESCRIPTION.search(test["description"])
        columns.append(ColumnDrift(
            column=name,
            column_type="numerical" if name in numerical else "categorical",
            score=float(scores[f"ValueDrift(column={name})"]),
            method=match["method"] if match else "unknown",
            threshold=float(match["threshold"]) if match else float("nan"),
            drifted=str(test["status"]).endswith("FAIL"),
        ))

    n_drifted = sum(c.drifted for c in columns)
    share = n_drifted / len(columns) if columns else 0.0
    result = DriftResult(
        n_reference=len(reference), n_current=len(current), n_columns=len(columns),
        n_drifted=n_drifted, drift_share=share, dataset_drift=share >= 0.5, columns=columns,
    )
    return result, snapshot


def save_outputs(result: DriftResult, snapshot, out_dir: Path | str, label: str) -> dict[str, Path]:
    """Write drift_report.html, summary.json and columns.csv; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"html": out_dir / "drift_report.html", "summary": out_dir / "summary.json",
             "columns": out_dir / "columns.csv"}
    snapshot.save_html(str(paths["html"]))
    paths["summary"].write_text(json.dumps({"label": label, **result.to_dict()}, indent=2) + "\n")
    pd.DataFrame([asdict(c) for c in result.columns]).to_csv(paths["columns"], index=False)
    return paths


# ----------------------------------------------------------------- batch monitoring

def gradual_drift_batches(reference: pd.DataFrame, current: pd.DataFrame, n_batches: int,
                          batch_size: int = 2000, seed: int = 42):
    """
    Simulate production traffic that drifts over time.

    The reference is split into a *baseline* (what the model was trained on)
    and a *holdout*. Batch i mixes i/(n-1) of its rows from `current` and the
    rest from the holdout, so batch 0 looks like training data and the last
    batch is pure `current`. Returns (baseline, [(mix_fraction, batch), ...]).
    """
    if n_batches < 2:
        raise InputError("--batches must be at least 2")
    rng = np.random.default_rng(seed)
    shuffled = reference.sample(frac=1.0, random_state=seed)
    baseline, holdout = shuffled.iloc[: len(shuffled) // 2], shuffled.iloc[len(shuffled) // 2:]
    batches = []
    for i in range(n_batches):
        frac = i / (n_batches - 1)
        n_cur = int(round(batch_size * frac))
        part_cur = current.sample(n=n_cur, replace=len(current) < n_cur, random_state=int(rng.integers(1e9)))
        part_ref = holdout.sample(n=batch_size - n_cur, replace=len(holdout) < batch_size - n_cur,
                                  random_state=int(rng.integers(1e9)))
        batches.append((frac, pd.concat([part_ref, part_cur])))
    return baseline, batches


def monitor_batches(baseline: pd.DataFrame, batches, numerical=None, categorical=None) -> pd.DataFrame:
    """Drift of every batch against the baseline, as one row per batch."""
    rows = []
    for i, (frac, batch) in enumerate(batches):
        result, _ = run_drift(baseline, batch, numerical, categorical)
        rows.append({"batch": i, "share_from_current": round(frac, 3), "n_rows": len(batch),
                     "n_drifted": result.n_drifted, "drift_share": round(result.drift_share, 4),
                     "drifted_columns": ";".join(result.drifted_columns)})
    return pd.DataFrame(rows)


def plot_trend(trend: pd.DataFrame, path: Path | str, gate: float | None = None) -> Path:
    """Line chart of drift share per batch (the monitoring dashboard view)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trend["batch"], trend["drift_share"], marker="o", label="share of drifted columns")
    if gate is not None:
        ax.axhline(gate, color="tab:red", linestyle="--", label=f"alert threshold ({gate:g})")
    ax.set_xlabel("batch (later = more data from the drifted population)")
    ax.set_ylabel("share of drifted columns")
    ax.set_ylim(0, 1)
    ax.set_xticks(trend["batch"])
    ax.set_title("Data drift over time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ CLI

def format_table(result: DriftResult) -> str:
    rows = sorted(result.columns, key=lambda c: (not c.drifted, -c.score))
    lines = [f"{'column':<16} {'type':<11} {'score':>8} {'thr':>5}  drift  method"]
    for c in rows:
        lines.append(f"{c.column:<16} {c.column_type:<11} {c.score:>8.4f} {c.threshold:>5.2f}  "
                     f"{'YES' if c.drifted else 'no':<5}  {c.method}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evidently data-drift monitor with a CI gate.")
    p.add_argument("--scenario", choices=SCENARIOS, default="education",
                   help="how to build reference/current from the Adult dataset (default: the notebook's)")
    p.add_argument("--reference", help="reference CSV (use with --current instead of a scenario)")
    p.add_argument("--current", help="current CSV")
    p.add_argument("--num-cols", nargs="+", help="numerical columns for CSV input (default: inferred)")
    p.add_argument("--cat-cols", nargs="+", help="categorical columns for CSV input (default: inferred)")
    p.add_argument("--fail-above", type=float, default=None, metavar="SHARE",
                   help="exit with code 1 if the share of drifted columns is above SHARE (0-1)")
    p.add_argument("--batches", type=int, default=0, help="also simulate N batches of gradual drift")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default: ./reports)")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="dataset cache folder")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.fail_above is not None and not 0 <= args.fail_above <= 1:
            raise InputError("--fail-above must be between 0 and 1")
        if bool(args.reference) != bool(args.current):
            raise InputError("--reference and --current must be given together")
        if args.reference:
            reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
            num, cat = args.num_cols, args.cat_cols
            if num is None and cat is None:
                num, cat = infer_schema(reference)
            label = f"{Path(args.reference).name} vs {Path(args.current).name}"
        else:
            reference, current = make_scenario(load_adult(args.cache), args.scenario, args.seed)
            num, cat = ADULT_NUMERICAL, ADULT_CATEGORICAL
            label = f"adult / {args.scenario}"
        result, snapshot = run_drift(reference, current, num, cat, with_summary=True)
    except (InputError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) / (args.scenario if not args.reference else "custom")
    paths = save_outputs(result, snapshot, out_dir, label)
    print(f"Drift report: {label}  (reference {result.n_reference} rows, current {result.n_current} rows)")
    print(format_table(result))
    print(f"\nDrifted columns: {result.n_drifted}/{result.n_columns} "
          f"(share {result.drift_share:.2f}); Evidently dataset drift: {result.dataset_drift}")
    print(f"Saved: {paths['html']}\n       {paths['summary']}\n       {paths['columns']}")

    if args.batches:
        try:
            baseline, batches = gradual_drift_batches(reference, current, args.batches, args.batch_size, args.seed)
        except InputError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        trend = monitor_batches(baseline, batches, num, cat)
        trend.to_csv(out_dir / "drift_trend.csv", index=False)
        chart = plot_trend(trend, out_dir / "drift_trend.png", args.fail_above)
        print("\nGradual drift over batches:")
        print(trend.drop(columns=["drifted_columns"]).to_string(index=False))
        print(f"Saved: {out_dir / 'drift_trend.csv'}\n       {chart}")

    if args.fail_above is not None and result.drift_share > args.fail_above:
        sys.stdout.flush()  # keep the table above the alert when both go to a terminal
        print(f"\nDRIFT ALERT: {result.drift_share:.2f} of columns drifted (> {args.fail_above:g})",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
