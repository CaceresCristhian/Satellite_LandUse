"""
Satellite Thermal Anomaly & Disturbance Characterization across Global Biomes.
Evaluates 2,299 NASA VIIRS 375m active satellite observations across 5 disturbance regimes:
1. Tropical Deforestation (Amazon)
2. Agricultural Conversion (Cerrado)
3. Industrial Flaring (Permian Basin)
4. Boreal Disturbance (North American Taiga)
5. Peatland Disturbance (Southeast Asia)
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "01_raw" / "nasa_viirs_satellite_disturbances.parquet"
PROC_DIR = BASE_DIR / "data" / "02_processed"
FIG_DIR = BASE_DIR / "paper" / "figures"
TAB_DIR = BASE_DIR / "paper" / "tables"
EXP_DIR = BASE_DIR / "experiments"

CLASS_NAMES = [
    "Tropical Deforestation",
    "Agricultural Conversion",
    "Industrial Flaring",
    "Boreal Disturbance",
    "Peatland Disturbance",
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SEEDS = [42, 43, 44, 45, 46]


def evaluate_model(clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred) * 100.0
    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0) * 100.0
    macro_rec = recall_score(y_te, y_pred, average="macro", zero_division=0) * 100.0
    macro_prec = precision_score(y_te, y_pred, average="macro", zero_division=0) * 100.0
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_rec": macro_rec,
        "macro_prec": macro_prec,
        "cm": cm,
        "y_pred": y_pred,
    }


def compute_physical_distributions(df):
    stats = {}
    for cname in CLASS_NAMES:
        sub = df[df["disturbance_class"] == cname]
        stats[cname] = {
            "n": len(sub),
            "ti4_med": float(sub["bright_ti4"].median()),
            "ti4_iqr": float(sub["bright_ti4"].quantile(0.75) - sub["bright_ti4"].quantile(0.25)),
            "ti5_med": float(sub["bright_ti5"].median()),
            "ti5_iqr": float(sub["bright_ti5"].quantile(0.75) - sub["bright_ti5"].quantile(0.25)),
            "delta_t_med": float(sub["delta_t"].median()),
            "delta_t_iqr": float(sub["delta_t"].quantile(0.75) - sub["delta_t"].quantile(0.25)),
            "frp_med": float(sub["frp"].median()),
            "frp_iqr": float(sub["frp"].quantile(0.75) - sub["frp"].quantile(0.25)),
            "saturated_count": int((sub["bright_ti4"] >= 367.0).sum()),
        }
    return stats


def run_pipeline():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NASA VIIRS satellite observations from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    # Filter physical radiometry validity bounds
    df = df[(df["bright_ti4"] >= 250.0) & (df["delta_t"] >= 0.0)].reset_index(drop=True)
    n_total = len(df)
    date_min = str(df["acq_date"].min())
    date_max = str(df["acq_date"].max())
    n_sat_total = int((df["bright_ti4"] >= 367.0).sum())
    print(f"Loaded {n_total:,} NASA satellite detections across {len(CLASS_NAMES)} disturbance classes.")
    print(f"Acquisition period: {date_min} to {date_max}. Saturated pixels (>=367K): {n_sat_total}")

    df["label"] = df["disturbance_class"].map(CLASS_MAP)
    features_rad = [
        "bright_ti4",
        "bright_ti5",
        "delta_t",
        "frp",
        "log_frp",
        "is_daytime",
    ]
    features_geom = features_rad + ["scan", "track"]

    # 1. Compute Physical Distributions for RQ3
    phys_stats = compute_physical_distributions(df)

    # 2. Spatial Tile Construction for Spatial Cross-Validation
    df["spatial_tile"] = (
        np.floor(df["latitude"] / 0.25).astype(int).astype(str) + "_" +
        np.floor(df["longitude"] / 0.25).astype(int).astype(str)
    )
    n_tiles = df["spatial_tile"].nunique()
    print(f"Constructed {n_tiles} spatial tiles (0.25-degree resolution) for block cross-validation.")

    # Spatial Block 5-Fold Cross-Validation on pure radiometry
    gkf = GroupKFold(n_splits=5)
    spatial_accs = []
    spatial_f1s = []
    for tr_idx, te_idx in gkf.split(df, df["label"], groups=df["spatial_tile"]):
        m_sp = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
        m_sp.fit(df.iloc[tr_idx][features_rad], df.iloc[tr_idx]["label"])
        y_sp_pred = m_sp.predict(df.iloc[te_idx][features_rad])
        spatial_accs.append(accuracy_score(df.iloc[te_idx]["label"], y_sp_pred) * 100.0)
        spatial_f1s.append(f1_score(df.iloc[te_idx]["label"], y_sp_pred, average="macro", zero_division=0) * 100.0)

    sp_acc_mean, sp_acc_std = float(np.mean(spatial_accs)), float(np.std(spatial_accs))
    sp_f1_mean, sp_f1_std = float(np.mean(spatial_f1s)), float(np.std(spatial_f1s))
    print(f"Spatial Block 5-Fold CV (HistGBM): Acc={sp_acc_mean:.1f}+/-{sp_acc_std:.1f}%, F1={sp_f1_mean:.1f}+/-{sp_f1_std:.1f}%")

    # 3. Multi-Seed Benchmark Evaluation on Random Split (5 Seeds)
    model_factories = {
        "Majority Baseline": lambda s: DummyClassifier(strategy="most_frequent"),
        "Coordinates-Only (Lat/Lon)": lambda s: HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=s),
        "Geometry-Only (Scan/Track)": lambda s: HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=s),
        "T_I5-Only (Ambient Temp)": lambda s: DecisionTreeClassifier(max_depth=4, random_state=s),
        "Logistic Regression (Balanced)": lambda s: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=s)),
        "Decision Tree (Depth 6)": lambda s: DecisionTreeClassifier(max_depth=6, random_state=s),
        "Multi-Layer Perceptron (MLP)": lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=s)),
        "Random Forest (150 trees)": lambda s: RandomForestClassifier(n_estimators=150, max_depth=10, random_state=s),
        "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=s),
        "HistGBM + Swath Geometry": lambda s: HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=s),
    }

    seed_runs = {name: {"acc": [], "macro_f1": [], "weighted_f1": [], "macro_rec": []} for name in model_factories}
    last_eval_results = {}
    rf_feature_importances = []

    last_split_sizes = (0, 0)
    for s in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df, df["label"], test_size=0.30, random_state=s, stratify=df["label"]
        )
        last_split_sizes = (len(y_tr), len(y_te))
        for name, factory in model_factories.items():
            clf = factory(s)
            if name == "Coordinates-Only (Lat/Lon)":
                res = evaluate_model(clf, X_tr[["latitude", "longitude"]].values, y_tr.values, X_te[["latitude", "longitude"]].values, y_te.values)
            elif name == "Geometry-Only (Scan/Track)":
                res = evaluate_model(clf, X_tr[["scan", "track"]].values, y_tr.values, X_te[["scan", "track"]].values, y_te.values)
            elif name == "T_I5-Only (Ambient Temp)":
                res = evaluate_model(clf, X_tr[["bright_ti5"]].values, y_tr.values, X_te[["bright_ti5"]].values, y_te.values)
            elif name == "HistGBM + Swath Geometry":
                res = evaluate_model(clf, X_tr[features_geom].values, y_tr.values, X_te[features_geom].values, y_te.values)
            else:
                res = evaluate_model(clf, X_tr[features_rad].values, y_tr.values, X_te[features_rad].values, y_te.values)

            seed_runs[name]["acc"].append(res["acc"])
            seed_runs[name]["macro_f1"].append(res["macro_f1"])
            seed_runs[name]["weighted_f1"].append(res["weighted_f1"])
            seed_runs[name]["macro_rec"].append(res["macro_rec"])

            if s == 42:
                last_eval_results[name] = res
                if name == "Random Forest (150 trees)":
                    rf_feature_importances = clf.feature_importances_

    # Aggregate Multi-Seed Summary
    benchmark_summary = {}
    print("\n================== SATELLITE DISTURBANCE BENCHMARK (5-SEED MEAN +/- SD) ==================")
    print(f"{'Model Architecture':<40} | {'Accuracy (%)':<16} | {'Macro-F1 (%)':<16} | {'Macro-Recall (%)':<16}")
    print("-" * 96)
    for name in model_factories:
        acc_m, acc_s = float(np.mean(seed_runs[name]["acc"])), float(np.std(seed_runs[name]["acc"]))
        f1_m, f1_s = float(np.mean(seed_runs[name]["macro_f1"])), float(np.std(seed_runs[name]["macro_f1"]))
        wf1_m, wf1_s = float(np.mean(seed_runs[name]["weighted_f1"])), float(np.std(seed_runs[name]["weighted_f1"]))
        rec_m, rec_s = float(np.mean(seed_runs[name]["macro_rec"])), float(np.std(seed_runs[name]["macro_rec"]))

        benchmark_summary[name] = {
            "acc_mean": acc_m, "acc_std": acc_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "wf1_mean": wf1_m, "wf1_std": wf1_s,
            "rec_mean": rec_m, "rec_std": rec_s,
        }
        print(f"{name:<40} | {acc_m:5.1f} +/- {acc_s:3.1f}% | {f1_m:5.1f} +/- {f1_s:3.1f}% | {rec_m:5.1f} +/- {rec_s:3.1f}%")

    # Feature Importance via Random Forest (Seed 42)
    rf_feat_imp = pd.Series(rf_feature_importances, index=features_rad).sort_values(ascending=False)

    # 4-Panel Publication Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Radiative Signature (I-4 Temp vs Delta-T)
    scatter_colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    for idx, cname in enumerate(CLASS_NAMES):
        sub = df[df["label"] == idx]
        ax1.scatter(
            sub["bright_ti4"],
            sub["delta_t"],
            c=scatter_colors[idx],
            label=cname,
            alpha=0.6,
            edgecolor="none",
            s=22,
        )
    ax1.axvline(367.0, color="grey", linestyle="--", alpha=0.7, label=r"I-4 Saturation Limit ($\approx 367\,$K)")
    ax1.set_xlabel(r"VIIRS I-4 Channel Brightness Temp $T_{I4}$ (K)", fontsize=10)
    ax1.set_ylabel(r"Thermal Contrast $\Delta T = T_{I4} - T_{I5}$ (K)", fontsize=10)
    ax1.set_title("(a) NASA VIIRS 375m Multi-Spectral Thermal Signatures", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    # Panel B: Feature Importance
    rf_feat_imp.plot(kind="barh", ax=ax2, color="#2b83ba", edgecolor="black")
    ax2.invert_yaxis()
    ax2.set_xlabel("Random Forest Gini Importance", fontsize=10)
    ax2.set_title("(b) Radiometric Feature Importance ($T_{I5}$ Dominance)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel C: Normalized Confusion Matrix (HistGBM Seed 42)
    res_hgb = last_eval_results["HistGradientBoosting"]
    cm_norm = res_hgb["cm"].astype("float") / res_hgb["cm"].sum(axis=1)[:, np.newaxis]
    im = ax3.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xticks(range(len(CLASS_NAMES)))
    ax3.set_yticks(range(len(CLASS_NAMES)))
    ax3.set_xticklabels([c.replace(" ", "\n") for c in CLASS_NAMES], fontsize=8)
    ax3.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax3.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax3.set_xlabel("Predicted Disturbance Class", fontsize=10)
    ax3.set_ylabel("True Class (Region-Derived)", fontsize=10)
    ax3.set_title("(c) Normalized Confusion Matrix (HistGBM Held-Out Test)", fontsize=11, fontweight="bold")

    # Panel D: Multi-Model Accuracy Comparison with Error Bars
    model_plot_keys = [
        "Majority Baseline",
        "T_I5-Only (Ambient Temp)",
        "Logistic Regression (Balanced)",
        "Decision Tree (Depth 6)",
        "Multi-Layer Perceptron (MLP)",
        "Random Forest (150 trees)",
        "HistGradientBoosting",
        "HistGBM + Swath Geometry",
    ]
    plot_labels = [
        "Majority",
        r"$T_{I5}$-Only",
        "Logistic (Bal.)",
        "Decision Tree",
        "MLP",
        "Random Forest",
        "HistGBM (Pure)",
        "HistGBM + Swath",
    ]
    plot_accs = [benchmark_summary[k]["acc_mean"] for k in model_plot_keys]
    plot_errs = [benchmark_summary[k]["acc_std"] for k in model_plot_keys]
    bar_colors = ["#999999", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd", "#386cb0", "#e41a1c"]

    bars = ax4.bar(range(len(plot_labels)), plot_accs, yerr=plot_errs, capsize=4, color=bar_colors, edgecolor="black")
    ax4.axhline(sp_acc_mean, color="navy", linestyle=":", linewidth=2, label=f"Spatial Block CV ({sp_acc_mean:.1f}%)")
    ax4.set_xticks(range(len(plot_labels)))
    ax4.set_xticklabels(plot_labels, rotation=25, ha="right", fontsize=8)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax4.set_title("(d) Disturbance Classification Benchmark (5-Seed Mean +/- SD)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig1_landuse_disturbance_benchmark.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {fig_path}")

    # Generate LaTeX Macros
    prec_flaring_hgb = res_hgb["cm"][2, 2] / res_hgb["cm"][:, 2].sum() * 100.0
    shortcut_boost = benchmark_summary["HistGBM + Swath Geometry"]["acc_mean"] - benchmark_summary["HistGradientBoosting"]["acc_mean"]
    spatial_leakage = benchmark_summary["HistGradientBoosting"]["acc_mean"] - sp_acc_mean

    n_train = last_split_sizes[0]
    n_test = last_split_sizes[1]

    with open(TAB_DIR / "macros.tex", "w") as f:
        f.write("% Auto-generated macros for NASA FIRMS VIIRS Disturbance Pipeline\n")
        f.write(f"\\newcommand{{\\numDetections}}{{{n_total:,}}}\n")
        f.write(f"\\newcommand{{\\numTrainDetections}}{{{n_train:,}}}\n")
        f.write(f"\\newcommand{{\\numTestDetections}}{{{n_test:,}}}\n")
        f.write(f"\\newcommand{{\\dateMin}}{{{date_min}}}\n")
        f.write(f"\\newcommand{{\\dateMax}}{{{date_max}}}\n")
        f.write(f"\\newcommand{{\\numSaturatedPixels}}{{{n_sat_total}}}\n")
        f.write(f"\\newcommand{{\\numSpatialTiles}}{{{n_tiles}}}\n")
        f.write(f"\\newcommand{{\\accDummy}}{{{benchmark_summary['Majority Baseline']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accCoords}}{{{benchmark_summary['Coordinates-Only (Lat/Lon)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accGeomOnly}}{{{benchmark_summary['Geometry-Only (Scan/Track)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accTiFiveOnly}}{{{benchmark_summary['T_I5-Only (Ambient Temp)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accLogReg}}{{{benchmark_summary['Logistic Regression (Balanced)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accDt}}{{{benchmark_summary['Decision Tree (Depth 6)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accMlp}}{{{benchmark_summary['Multi-Layer Perceptron (MLP)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accRf}}{{{benchmark_summary['Random Forest (150 trees)']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accHgb}}{{{benchmark_summary['HistGradientBoosting']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\accSwathGeom}}{{{benchmark_summary['HistGBM + Swath Geometry']['acc_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\shortcutBoost}}{{{shortcut_boost:.1f}}}\n")
        f.write(f"\\newcommand{{\\accSpatialBlockCv}}{{{sp_acc_mean:.1f}}}\n")
        f.write(f"\\newcommand{{\\accSpatialBlockCvStd}}{{{sp_acc_std:.1f}}}\n")
        f.write(f"\\newcommand{{\\spatialLeakageInflation}}{{{spatial_leakage:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneDummy}}{{{benchmark_summary['Majority Baseline']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneLogReg}}{{{benchmark_summary['Logistic Regression (Balanced)']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneDt}}{{{benchmark_summary['Decision Tree (Depth 6)']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneMlp}}{{{benchmark_summary['Multi-Layer Perceptron (MLP)']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneRf}}{{{benchmark_summary['Random Forest (150 trees)']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneHgb}}{{{benchmark_summary['HistGradientBoosting']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneGeomOnly}}{{{benchmark_summary['Geometry-Only (Scan/Track)']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneSwathGeom}}{{{benchmark_summary['HistGBM + Swath Geometry']['f1_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recHgb}}{{{benchmark_summary['HistGradientBoosting']['rec_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\topFeature}}{{{rf_feat_imp.index[0]}}}\n")
        f.write(f"\\newcommand{{\\featImpTiFive}}{{{rf_feat_imp['bright_ti5']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\featImpDeltaT}}{{{rf_feat_imp['delta_t']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\featImpTiFour}}{{{rf_feat_imp['bright_ti4']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\featImpLogFrp}}{{{rf_feat_imp['log_frp']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\featImpFrp}}{{{rf_feat_imp['frp']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\featImpDaytime}}{{{rf_feat_imp['is_daytime']*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\recBorealHgb}}{{{cm_norm[3, 3]*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\recPeatlandHgb}}{{{cm_norm[4, 4]*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\recFlaringHgb}}{{{cm_norm[2, 2]*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\precFlaringHgb}}{{{prec_flaring_hgb:.1f}}}\n")
        # Physical stats macros
        f.write(f"\\newcommand{{\\flaringDeltaTMed}}{{{phys_stats['Industrial Flaring']['delta_t_med']:.1f}}}\n")
        f.write(f"\\newcommand{{\\flaringTiFourMed}}{{{phys_stats['Industrial Flaring']['ti4_med']:.1f}}}\n")
        f.write(f"\\newcommand{{\\flaringFrpMed}}{{{phys_stats['Industrial Flaring']['frp_med']:.2f}}}\n")
        f.write(f"\\newcommand{{\\borealDeltaTMed}}{{{phys_stats['Boreal Disturbance']['delta_t_med']:.1f}}}\n")
        f.write(f"\\newcommand{{\\borealTiFiveMed}}{{{phys_stats['Boreal Disturbance']['ti5_med']:.1f}}}\n")

    # Generate Benchmark Table with Mean +/- SD
    with open(TAB_DIR / "tab_benchmark.tex", "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Disturbance Characterization Benchmark on NASA VIIRS Detections ($N = \numDetections{}$, 5-Seed Mean $\pm$ SD).}" + "\n")
        f.write(r"\label{tab:satellite_benchmark}" + "\n")
        f.write(r"\resizebox{\columnwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Model / Configuration} & \textbf{Accuracy (\%)} & \textbf{Macro-F1 (\%)} & \textbf{Weighted-F1 (\%)} & \textbf{Macro-Recall (\%)} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{5}{l}{\textit{Reference and Control Baselines}} \\" + "\n")
        for k in ["Majority Baseline", "Coordinates-Only (Lat/Lon)", "Geometry-Only (Scan/Track)", "T_I5-Only (Ambient Temp)"]:
            b = benchmark_summary[k]
            label = r"$T_{I5}$-Only (Ambient Temp)" if k == "T_I5-Only (Ambient Temp)" else k
            f.write(f"{label} & {b['acc_mean']:.1f} $\\pm$ {b['acc_std']:.1f} & {b['f1_mean']:.1f} $\\pm$ {b['f1_std']:.1f} & {b['wf1_mean']:.1f} $\\pm$ {b['wf1_std']:.1f} & {b['rec_mean']:.1f} $\\pm$ {b['rec_std']:.1f} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{5}{l}{\textit{Pure Radiometric Classifiers}} \\" + "\n")
        for k in ["Logistic Regression (Balanced)", "Decision Tree (Depth 6)", "Multi-Layer Perceptron (MLP)", "Random Forest (150 trees)", "HistGradientBoosting"]:
            b = benchmark_summary[k]
            f.write(f"{k} & {b['acc_mean']:.1f} $\\pm$ {b['acc_std']:.1f} & {b['f1_mean']:.1f} $\\pm$ {b['f1_std']:.1f} & {b['wf1_mean']:.1f} $\\pm$ {b['wf1_std']:.1f} & {b['rec_mean']:.1f} $\\pm$ {b['rec_std']:.1f} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{5}{l}{\textit{Orbital Swath Geometry Shortcut Ablation}} \\" + "\n")
        b = benchmark_summary["HistGBM + Swath Geometry"]
        f.write(f"HistGBM + Swath (Scan/Track) & \\textbf{{{b['acc_mean']:.1f} $\\pm$ {b['acc_std']:.1f}}} & \\textbf{{{b['f1_mean']:.1f} $\\pm$ {b['f1_std']:.1f}}} & \\textbf{{{b['wf1_mean']:.1f} $\\pm$ {b['wf1_std']:.1f}}} & \\textbf{{{b['rec_mean']:.1f} $\\pm$ {b['rec_std']:.1f}}} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"Spatial Block 5-Fold CV (HistGBM) & {sp_acc_mean:.1f} $\\pm$ {sp_acc_std:.1f} & {sp_f1_mean:.1f} $\\pm$ {sp_f1_std:.1f} & --- & --- \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

    # Generate Physical Distributions Table for RQ3
    with open(TAB_DIR / "tab_physical_distributions.tex", "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Empirical Thermal Radiometry Distributions Across Disturbance Classes (Median [IQR]).}" + "\n")
        f.write(r"\label{tab:physical_distributions}" + "\n")
        f.write(r"\resizebox{\columnwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Disturbance Class} & $N$ & $T_{I4}$ (K) & $T_{I5}$ (K) & $\Delta T$ (K) & FRP (MW) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for cname in CLASS_NAMES:
            p = phys_stats[cname]
            f.write(f"{cname} & {p['n']:,} & {p['ti4_med']:.1f} [{p['ti4_iqr']:.1f}] & {p['ti5_med']:.1f} [{p['ti5_iqr']:.1f}] & {p['delta_t_med']:.1f} [{p['delta_t_iqr']:.1f}] & {p['frp_med']:.2f} [{p['frp_iqr']:.2f}] \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

    # Save complete JSON results
    full_results = {
        "dataset": {
            "n_total": n_total,
            "date_min": date_min,
            "date_max": date_max,
            "n_sat_total": n_sat_total,
            "n_spatial_tiles": n_tiles,
        },
        "spatial_block_cv": {
            "acc_mean": sp_acc_mean, "acc_std": sp_acc_std,
            "f1_mean": sp_f1_mean, "f1_std": sp_f1_std,
        },
        "benchmark_summary": benchmark_summary,
        "physical_distributions": phys_stats,
        "feature_importances": rf_feat_imp.to_dict(),
    }
    with open(EXP_DIR / "results_summary.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("Saved all publication artifacts, macros, and results JSON successfully.")


if __name__ == "__main__":
    run_pipeline()
