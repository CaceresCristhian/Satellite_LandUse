"""
Unit tests for NASA VIIRS Satellite Disturbance Intelligence Pipeline.
Tests authentic satellite data integrity, physical radiometry bounds,
multi-class mapping, stratified split balance, spatial tile disjointness,
macro consistency with results JSON, and working-tree security scanning.
"""

import sys
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train_landuse_transformer import (
    CLASS_MAP,
    CLASS_NAMES,
)

DATA_PATH = BASE_DIR / "data" / "01_raw" / "nasa_viirs_satellite_disturbances.parquet"
FIG_PATH = BASE_DIR / "paper" / "figures" / "fig1_landuse_disturbance_benchmark.png"
MACROS_PATH = BASE_DIR / "paper" / "tables" / "macros.tex"
TAB_PATH = BASE_DIR / "paper" / "tables" / "tab_benchmark.tex"
PHYS_TAB_PATH = BASE_DIR / "paper" / "tables" / "tab_physical_distributions.tex"
TEX_PATH = BASE_DIR / "paper" / "main.tex"
RESULTS_JSON_PATH = BASE_DIR / "experiments" / "results_summary.json"


def test_nasa_firms_data_integrity():
    """Verify authentic NASA VIIRS satellite observations have >=2,000 rows and 5 classes."""
    assert DATA_PATH.exists(), f"NASA dataset missing at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    assert len(df) >= 2000, f"Expected >= 2,000 satellite records, got {len(df)}"
    assert set(df["disturbance_class"].unique()) == set(CLASS_NAMES)
    # Physical sensor sanity checks
    assert (df["bright_ti4"] >= 200.0).all(), "Raw VIIRS I-4 brightness temp must be >= 200K"
    assert (df["bright_ti5"] >= 200.0).all(), "Raw VIIRS I-5 brightness temp must be >= 200K"
    assert (df["frp"] >= 0.0).all(), "Fire Radiative Power (MW) must be non-negative"


def test_physical_filter_bounds():
    """Verify the production filter removes records below 250K I-4 or negative delta_t."""
    df = pd.read_parquet(DATA_PATH)
    filtered = df[(df["bright_ti4"] >= 250.0) & (df["delta_t"] >= 0.0)].reset_index(drop=True)
    assert (filtered["bright_ti4"] >= 250.0).all()
    assert (filtered["delta_t"] >= 0.0).all()
    assert len(filtered) == 2299, f"Expected exactly 2,299 filtered detections, got {len(filtered)}"


def test_thermal_contrast_calculation():
    """Verify differential thermal contrast delta_t = bright_ti4 - bright_ti5."""
    df = pd.read_parquet(DATA_PATH)
    expected_delta = df["bright_ti4"] - df["bright_ti5"]
    np.testing.assert_allclose(df["delta_t"].values, expected_delta.values, rtol=1e-5)


def test_class_mapping_completeness():
    """Ensure all 5 disturbance classes map to discrete contiguous integer labels [0..4]."""
    assert len(CLASS_MAP) == 5
    assert set(CLASS_MAP.values()) == {0, 1, 2, 3, 4}
    df = pd.read_parquet(DATA_PATH)
    mapped = df["disturbance_class"].map(CLASS_MAP)
    assert mapped.notnull().all(), "All classes must be mapped without NaNs"


def test_spatial_block_cv_disjointness():
    """Verify that across 5-fold spatial block CV, no 0.25-degree tile appears in both train and test partitions."""
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["bright_ti4"] >= 250.0) & (df["delta_t"] >= 0.0)].reset_index(drop=True)
    tiles = (
        np.floor(df["latitude"] / 0.25).astype(int).astype(str) + "_" +
        np.floor(df["longitude"] / 0.25).astype(int).astype(str)
    )
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=tiles)):
        tr_tiles = set(tiles.iloc[tr_idx])
        te_tiles = set(tiles.iloc[te_idx])
        overlap = tr_tiles.intersection(te_tiles)
        assert len(overlap) == 0, f"Spatial leakage in fold {fold}: {len(overlap)} overlapping tiles!"


def test_macros_consistency_with_pipeline_results():
    """Verify all LaTeX macros match the computed metrics from experiments/results_summary.json."""
    assert RESULTS_JSON_PATH.exists(), "results_summary.json must exist"
    assert MACROS_PATH.exists(), "macros.tex must exist"

    with open(RESULTS_JSON_PATH, "r") as f:
        data = json.load(f)
    tex = MACROS_PATH.read_text(encoding="utf-8")

    def get_macro(name):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.+?)\}", tex)
        assert m, f"Macro \\{name} not found in macros.tex"
        return m.group(1).replace(",", "")

    # Compare sample size and tiles
    assert int(get_macro("numDetections")) == data["dataset"]["n_total"]
    assert int(get_macro("numSpatialTiles")) == data["dataset"]["n_spatial_tiles"]
    assert int(get_macro("numSaturatedPixels")) == data["dataset"]["n_sat_total"]

    # Compare key model accuracies
    rf_acc = data["benchmark_summary"]["Random Forest (150 trees)"]["acc_mean"]
    assert abs(float(get_macro("accRf")) - rf_acc) < 0.15

    swath_acc = data["benchmark_summary"]["HistGBM + Swath Geometry"]["acc_mean"]
    assert abs(float(get_macro("accSwathGeom")) - swath_acc) < 0.15

    sp_acc = data["spatial_block_cv"]["acc_mean"]
    assert abs(float(get_macro("accSpatialBlockCv")) - sp_acc) < 0.15


def test_publication_artifacts_integrity():
    """Check existence, file sizes, and LaTeX syntax validity."""
    assert FIG_PATH.exists() and FIG_PATH.stat().st_size > 20000, "Figure 1 must exist and be >20KB"
    assert MACROS_PATH.exists() and MACROS_PATH.stat().st_size > 100, "Macros TeX must exist"
    assert TAB_PATH.exists() and TAB_PATH.stat().st_size > 100, "Benchmark table must exist"
    assert PHYS_TAB_PATH.exists() and PHYS_TAB_PATH.stat().st_size > 100, "Physical distributions table must exist"
    assert TEX_PATH.exists(), "main.tex must exist"

    tex_content = TEX_PATH.read_text(encoding="utf-8")
    assert "\\begin{document}" in tex_content
    assert "\\end{document}" in tex_content
    # Check no unescaped ampersands in titles or section headers
    for line in tex_content.splitlines():
        if line.strip().startswith(("\\section{", "\\subsection{", "\\title{")):
            clean_line = line.replace("\\&", "")
            assert "&" not in clean_line, f"Unescaped '&' found in section/title: {line}"


def test_security_secrets_scan():
    """Verify that no API keys, private tokens, or credentials exist in the repository source."""
    forbidden_files = [".env", ".env.local", "mcp_config.json", "Copernicus.txt"]
    for f in forbidden_files:
        assert not (BASE_DIR / f).exists(), f"Security violation: {f} found in project directory!"

    suspicious_patterns = [
        re.compile(r"-----BEGIN [A-Z]+ PRIVATE KEY-----"),
        re.compile(r"ghp_[A-Za-z0-9_]{36}"),
        re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        re.compile(r"MAP_KEY\s*=\s*['\"][0-9a-fA-F]{32}['\"]"),
    ]
    for ext in ["*.py", "*.md", "*.tex", "*.sh"]:
        for file_path in BASE_DIR.rglob(ext):
            if "test_pipeline.py" in str(file_path):
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in suspicious_patterns:
                assert not pattern.search(content), f"Secret pattern matched in {file_path.name}"
