"""The notebook itself, and the real Adult-dataset results (network on first run)."""
import json
import re
import shutil
from pathlib import Path

import pytest

import drift_monitor as dm

LAB_DIR = Path(dm.__file__).resolve().parent
REPO = LAB_DIR.parents[1]
NOTEBOOK = LAB_DIR / "Lab1.ipynb"

# Result of the ORIGINAL notebook (docs/evidence/original_behaviour.txt).
ORIGINAL_DRIFTED = {"education", "education-num", "age", "occupation", "native-country"}
ORIGINAL_SCORES = {"education-num": 0.617697, "age": 0.185347, "education": 0.832555, "class": 0.034295}


def cells():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]]


# ------------------------------------------------------------ notebook (no network)

def test_no_hard_coded_token_anywhere_in_repo():
    # Long base64-looking strings passed as token= would be a leaked credential.
    pattern = re.compile(r"""token\s*=\s*["'][A-Za-z0-9+/=]{40,}["']""")
    for path in REPO.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".ipynb", ".md", ".txt", ".yml", ".ini"} \
                and ".venv" not in path.parts and "data_cache" not in path.parts:
            assert not pattern.search(path.read_text(errors="ignore")), path


def test_cloud_workspace_is_optional_and_reads_env():
    source = cells()[3]
    assert 'os.environ.get("EVIDENTLY_API_TOKEN")' in source
    assert "if token else None" in source


def test_install_line_pins_pandas_below_3():
    assert '"pandas<3"' in cells()[0]


def test_original_analysis_cells_are_unchanged():
    c = cells()
    assert 'adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]' in c[5]
    assert 'numerical_columns=["education-num", "age", "capital-gain", "hours-per-week", "capital-loss", "fnlwgt"]' in c[6]
    assert "report = Report([\n    DataDriftPreset() \n])" in c[9]
    assert "my_eval = report.run(eval_data_1, eval_data_2)" in c[9]


def test_notebook_saves_reports_and_uses_summary_preset():
    last = cells()[-1]
    assert "save_html('reports/notebook_drift_report.html')" in last
    assert "DataSummaryPreset()" in last


def test_notebook_schema_matches_monitor_schema():
    source = cells()[6]
    for col in dm.ADULT_NUMERICAL + dm.ADULT_CATEGORICAL:
        assert f'"{col}"' in source


# ------------------------------------------------------- real Adult data (network)

@pytest.fixture(scope="module")
def adult():
    return dm.load_adult(dm.DEFAULT_CACHE)


@pytest.mark.network
def test_education_scenario_reproduces_original_notebook(adult):
    reference, current = dm.make_scenario(adult, "education")
    assert (len(reference), len(current)) == (14155, 34687)
    result, _ = dm.run_drift(reference, current)
    assert set(result.drifted_columns) == ORIGINAL_DRIFTED
    assert result.n_drifted == 5 and result.n_columns == 15
    scores = {c.column: c.score for c in result.columns}
    for col, value in ORIGINAL_SCORES.items():
        assert scores[col] == pytest.approx(value, abs=1e-6)


@pytest.mark.network
def test_random_split_control_has_no_drift(adult):
    result, _ = dm.run_drift(*dm.make_scenario(adult, "random", seed=42))
    assert result.n_drifted == 0


@pytest.mark.network
def test_age_shift_flags_only_age(adult):
    result, _ = dm.run_drift(*dm.make_scenario(adult, "age_shift", seed=42))
    assert result.drifted_columns == ["age"]


@pytest.mark.network
def test_cli_default_scenario_with_gate(tmp_path, capsys):
    code = dm.main(["--out", str(tmp_path), "--fail-above", "0.3"])
    assert code == 1  # 5/15 = 0.33 > 0.3
    assert "Drifted columns: 5/15" in capsys.readouterr().out


@pytest.mark.network
def test_notebook_executes_end_to_end(tmp_path, adult):
    nbformat = pytest.importorskip("nbformat")
    from nbclient import NotebookClient

    shutil.copy(NOTEBOOK, tmp_path / "Lab1.ipynb")
    shutil.copytree(dm.DEFAULT_CACHE, tmp_path / "data_cache")  # reuse the download
    nb = nbformat.read(tmp_path / "Lab1.ipynb", as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert (tmp_path / "reports" / "notebook_drift_report.html").exists()
    assert (tmp_path / "reports" / "notebook_summary_report.html").exists()
    printed = "".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []))
    assert "Evidently Cloud: skipped" in printed
