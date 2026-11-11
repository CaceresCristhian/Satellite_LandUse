# Satellite Thermal Radiometry Disturbance Regime Characterization across Global Frontiers

## Overview
This repository implements a multi-class environmental disturbance characterization framework evaluating spaceborne radiances acquired by the Visible Infrared Imaging Radiometer Suite (VIIRS 375m) aboard the NASA/NOAA Suomi-NPP satellite.

## Satellite Data Provenance
- **Source**: NASA Earth Observing System Data and Information System (EOSDIS) Fire Information for Resource Management System (FIRMS).
- **Sensor**: VIIRS 375m active thermal anomaly product (`VIIRS_SNPP_NRT`).
- **Acquisition Window**: 2026-10-17 to 2026-10-21 (5-day Near Real-Time snapshot of record).
- **Sample Count**: 2,299 satellite detections across 5 contrasting global disturbance regimes:
  1. **Tropical Deforestation** (Brazilian Amazon arc, Mato Grosso & Rondônia): 391 detections
  2. **Agricultural Conversion** (Brazilian Cerrado, Maranhão & Tocantins): 520 detections
  3. **Industrial Petrochemical Flaring** (Texas Permian Basin): 157 detections
  4. **Boreal Disturbance** (Canadian Boreal Taiga, Alberta & NWT): 431 detections
  5. **Tropical Peatland Disturbance** (Sumatra, Indonesia): 800 detections
- **Data Filtering**: Excludes unphysical observations ($T_{I4} < 250\text{ K}$ or $\Delta T < 0\text{ K}$). Exactly 43 detections hit the operational sensor saturation limit ($T_{I4} \ge 367\text{ K}$).
- **Spatial Tiling**: Partitioned into 464 spatial tiles ($0.25^\circ \times 0.25^\circ$, $\approx 28\text{ km}$) for spatial block cross-validation.

## Multi-Spectral Radiometry Features
- $T_{I4}$: Mid-infrared 375m I-4 brightness temperature (Kelvin)
- $T_{I5}$: Thermal-infrared 375m I-5 brightness temperature (Kelvin, capturing regional ambient baseline)
- $\Delta T$: Differential thermal contrast ($T_{I4} - T_{I5}$)
- $\text{FRP}$: Fire Radiative Power (MW) and $\ln(1 + \text{FRP})$
- Solar illumination: Binary day/night flag

## Empirical Benchmark Results (5-Seed Mean $\pm$ SD)
| Model / Configuration | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Macro-Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Reference & Control Baselines** | | | | |
| Majority Baseline | 34.8 $\pm$ 0.0 | 10.3 $\pm$ 0.0 | 18.0 $\pm$ 0.0 | 20.0 $\pm$ 0.0 |
| Coordinates-Only (Lat/Lon) | 99.8 $\pm$ 0.1 | 99.8 $\pm$ 0.1 | 99.8 $\pm$ 0.1 | 99.8 $\pm$ 0.1 |
| $T_{I5}$-Only (Ambient Baseline) | 52.3 $\pm$ 0.7 | 39.9 $\pm$ 2.7 | 47.3 $\pm$ 1.2 | 42.7 $\pm$ 1.4 |
| **Pure Radiometric Classifiers** | | | | |
| Logistic Regression (Class-Balanced) | 55.4 $\pm$ 0.7 | 54.4 $\pm$ 0.9 | 55.5 $\pm$ 0.5 | 59.7 $\pm$ 1.2 |
| Decision Tree (Depth 6) | 61.6 $\pm$ 1.6 | 57.0 $\pm$ 1.0 | 58.7 $\pm$ 0.7 | 57.7 $\pm$ 1.5 |
| Multi-Layer Perceptron (MLP) | 61.7 $\pm$ 0.5 | 57.8 $\pm$ 0.7 | 59.1 $\pm$ 0.3 | 59.4 $\pm$ 0.9 |
| **Random Forest (150 trees)** | **63.1 $\pm$ 0.6** | **60.4 $\pm$ 1.4** | **61.6 $\pm$ 0.6** | **59.8 $\pm$ 1.8** |
| HistGradientBoosting (HistGBM) | 60.3 $\pm$ 1.5 | 58.5 $\pm$ 2.0 | 59.5 $\pm$ 1.5 | 58.2 $\pm$ 2.1 |
| **Orbital Swath Shortcut Ablation** | | | | |
| Geometry-Only (Scan/Track) | 53.0 $\pm$ 1.0 | 47.3 $\pm$ 1.9 | 51.8 $\pm$ 1.1 | 45.7 $\pm$ 1.5 |
| HistGBM + Swath Geometry (scan, track) | **83.4 $\pm$ 0.6** | **81.3 $\pm$ 0.8** | **83.3 $\pm$ 0.6** | **80.8 $\pm$ 1.1** |
| **Spatial Cross-Validation** | | | | |
| Spatial Block 5-Fold CV (HistGBM) | 55.5 $\pm$ 5.8 | 53.7 $\pm$ 5.2 | --- | --- |

### Key Findings
1. **Shortcut Learning Evaluation**: Incorporating satellite swath footprint geometry (\texttt{scan}, \texttt{track}) inflates apparent classification accuracy by **+23.1 percentage points** (from 60.3% to 83.4%), empirically demonstrating that raw swath models learn orbital track shortcuts rather than combustion physics.
2. **Spatial Autocorrelation Generalization**: When evaluated under 5-fold spatial block cross-validation across 464 independent 0.25° tiles, pure radiometric accuracy drops to 53.8% $\pm$ 5.5%, quantifying the 4.8 point optimism of random pixel splits.
3. **Physical Distinction**: Background ambient temperature ($T_{I5}$, 36.6%) and thermal contrast ($\Delta T$, 16.9%) provide the primary discriminatory signals. Industrial flaring exhibits low median $\Delta T$ (20.5 K) and low FRP (0.85 MW) within 375m pixels due to localized dissipation across the background, explaining its 27.7% confusion with peatland smoldering.

## Execution
The committed parquet snapshot (`data/01_raw/nasa_viirs_satellite_disturbances.parquet`) is the frozen dataset of record (October 17–21, 2026 NRT acquisition).

```bash
# 1. Run the test suite (8/8 passing)
pytest -v tests/test_pipeline.py

# 2. Train, evaluate, and generate all benchmark artifacts
python src/train_landuse_transformer.py

# 3. Compile the manuscript PDF
tectonic paper/main.tex --outdir paper/
```
