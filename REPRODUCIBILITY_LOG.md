# Reproducibility Log: Satellite Thermal Disturbance Characterization

**Manuscript Title**: Satellite Thermal Anomaly and Disturbance Regime Characterization across Global Forest and Agricultural Frontiers  
**Authors**: Cristhian David Cáceres Mateus, Talha Ali Khan (corresponding author: talhaali.khan@ue-germany.de)  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  
**Evaluation Cohort**: 2,299 NASA VIIRS 375m active fire records (Train: 1,609, Held-Out Test: 690 across 5 seeds)  
**Log Updated**: October 2026 (Round 6 Verification)  

---

## 1. Reproduction Steps

To execute the data ingestion, training pipeline, figure/macro generation, and compile the manuscript:

```bash
# 1. Environment requirements
pip install numpy pandas scikit-learn matplotlib pyarrow pytest requests

# 2. Run the test suite (8/8 passing)
pytest -v Satellite_LandUse_Change_Intelligence/tests

# 3. Execute the model training and benchmark pipeline (generates tables, figures, macros, results.json)
python Satellite_LandUse_Change_Intelligence/src/train_landuse_transformer.py

# 4. Compile the manuscript PDF (0 fatal errors)
tectonic Satellite_LandUse_Change_Intelligence/paper/main.tex --outdir Satellite_LandUse_Change_Intelligence/paper/
```

---

## 2. Data Sources and Provenance

- **Data Source**: NASA Fire Information for Resource Management System (FIRMS) for the VIIRS sensor aboard Suomi-NPP.
- **Sensor Attributes**: 375m I-band radiometry ($T_{I4}$ mid-infrared $3.74\,\mu\text{m}$, $T_{I5}$ thermal infrared $11.45\,\mu\text{m}$, Fire Radiative Power $\text{FRP}$, solar day/night indicator).
- **Acquisition Range**: 2026-10-17 to 2026-10-21 (5-day Near Real-Time snapshot of record).
- **Disturbance Biomes Sampled**:
  1. *Tropical Deforestation*: Brazilian Amazon (Mato Grosso and Rondônia) - 391 detections
  2. *Agricultural Conversion*: Brazilian Cerrado (Maranhão and Tocantins) - 520 detections
  3. *Industrial Flaring*: Texas Permian Basin - 157 detections
  4. *Boreal Disturbance*: Canadian Boreal taiga (Alberta and Northwest Territories) - 431 detections
  5. *Peatland Disturbance*: Tropical peat swamp clearing (Sumatra, Indonesia) - 800 detections
- **Data Filtering & Quality**: Exactly 2,299 satellite detections meet physical bounds ($T_{I4} \ge 250\text{ K}$, $\Delta T \ge 0\text{ K}$). Exactly 43 detections reach the sensor saturation ceiling ($T_{I4} \ge 367\text{ K}$).
- **Spatial Tiling**: Partitioned into 464 independent spatial tiles ($0.25^\circ \times 0.25^\circ$, $\approx 28\text{ km}$).

---

## 3. Methodological Protocol and Shortcut Mitigation

1. **Pure Radiometric Formulation**:
   - Swath footprint geometry (`scan`, `track`) is deliberately excluded from default features to avoid orbital scan-angle overfitting.
   - The feature space comprises: $T_{I4}, T_{I5}, \Delta T = T_{I4} - T_{I5}, \text{FRP}, \ln(1 + \text{FRP})$, and solar illumination.
2. **Swath Geometry Shortcut Ablation**:
   - An explicit ablation shows that adding `scan` and `track` inflates classification accuracy from 60.3% to 83.4% (+23.1 percentage points), proving that raw swath models learn orbital track footprints rather than combustion physics.
3. **Spatial Block Cross-Validation**:
   - 5-fold `GroupKFold` across 464 spatial tiles yields 53.8% $\pm$ 5.5% accuracy, quantifying the ~6.5 percentage point optimism of random pixel splits due to spatial autocorrelation.
4. **Physical Motivation & Distributions**:
   - Differential thermal contrast ($\Delta T$) is evaluated alongside ambient background temperature ($T_{I5}$).
   - Industrial flaring exhibits median $\Delta T = 20.5\text{ K}$ and median $\text{FRP} = 0.85\text{ MW}$, explaining its 27.7% confusion with peatland smoldering.
   - Boreal taiga fires exhibit median $\Delta T = 51.6\text{ K}$, driven by cold sub-arctic background temperatures ($T_{I5} = 277.9\text{ K}$).
5. **Benchmark Suite (5-Seed Mean $\pm$ SD)**:
   - Majority Baseline: 34.8% $\pm$ 0.0%
   - Coordinates-Only (Lat/Lon): 99.8% $\pm$ 0.1%
   - $T_{I5}$-Only (Ambient Baseline): 52.3% $\pm$ 0.7%
   - Logistic Regression (Class-Balanced): 55.4% $\pm$ 0.7%
   - Decision Tree (Depth 6): 61.6% $\pm$ 1.6%
   - Multi-Layer Perceptron (MLP): 61.7% $\pm$ 0.5%
   - Random Forest (150 trees): 63.1% $\pm$ 0.6%
   - HistGradientBoosting: 60.3% $\pm$ 1.5%
   - HistGBM + Swath Geometry: 83.4% $\pm$ 0.6%

---

## 4. Current Verification Status

- **Pipeline Execution**: `src/train_landuse_transformer.py` runs end-to-end and exports `tables/macros.tex`, `tables/tab_benchmark.tex`, `tables/tab_physical_distributions.tex`, `figures/fig1_landuse_disturbance_benchmark.png`, and `experiments/results_summary.json`.
- **Test Suite**: 8/8 unit and property tests passing (`pytest -v Satellite_LandUse_Change_Intelligence/tests`).
- **LaTeX Compilation**: Compiles via Tectonic with 0 fatal errors.
- **Prose Quality**: 0 errors, 0 warnings with Vale academic prose linter.
- **Remaining Open Items (Long-Term Data Scope)**:
  - Querying multi-region bounding boxes across continents to separate class signatures from single-region biomes.
  - Ingesting dedicated VIIRS Nightfire catalogs for petrochemical flares.
