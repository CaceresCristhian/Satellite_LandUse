"""
Download authentic satellite observations from NASA FIRMS API (VIIRS 375m active thermal detections).
Ingests genuine satellite measurements across 5 contrasting global disturbance regimes:
1. Tropical Deforestation (Brazilian Amazon Arc)
2. Agricultural Conversion & Biomass Clearing (Brazilian Cerrado)
3. Industrial Petrochemical Gas Flaring (Texas Permian Basin)
4. Boreal Forest Disturbance (North American Taiga / Canada)
5. Tropical Peatland Disturbance (Southeast Asia / Sumatra)
"""

import os
from pathlib import Path
import io
import urllib.request
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "01_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DIR / "nasa_viirs_satellite_disturbances.parquet"


def get_firms_key():
    """Retrieve NASA FIRMS MAP_KEY securely from environment variable or .env file."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        return key.strip()
    # Search project-level .env, then parent workspace .env
    for candidate_env in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if candidate_env.exists():
            for line in candidate_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("FIRMS_MAP_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("FIRMS_MAP_KEY not found. Set it in os.environ or in the project .env file.")

REGIONS = [
    {
        "region_id": "Amazon_Deforestation",
        "biome": "Tropical Moist Forest",
        "disturbance_class": "Tropical Deforestation",
        "bbox": "-62,-12,-53,-8",
        "days": 5,
        "max_samples": 800,
    },
    {
        "region_id": "Cerrado_Agricultural",
        "biome": "Wooded Savanna",
        "disturbance_class": "Agricultural Conversion",
        "bbox": "-49,-11,-44,-6",
        "days": 5,
        "max_samples": 800,
    },
    {
        "region_id": "Permian_Industrial",
        "biome": "Semi-Arid Basin",
        "disturbance_class": "Industrial Flaring",
        "bbox": "-104.5,31.0,-101.5,33.0",
        "days": 5,
        "max_samples": 800,
    },
    {
        "region_id": "Boreal_Wildfire",
        "biome": "Boreal Conifer Forest",
        "disturbance_class": "Boreal Disturbance",
        "bbox": "-125,50,-105,62",
        "days": 5,
        "max_samples": 800,
    },
    {
        "region_id": "Sumatra_Peatland",
        "biome": "Tropical Peat Swamp",
        "disturbance_class": "Peatland Disturbance",
        "bbox": "101,-4,106,0",
        "days": 5,
        "max_samples": 800,
    },
]


def download_nasa_firms_data():
    firms_key = get_firms_key()
    if not firms_key.isalnum():
        raise ValueError("Invalid FIRMS_MAP_KEY format: must be alphanumeric.")
    print("Downloading authentic NASA VIIRS satellite observations using configured MAP_KEY...")
    all_dfs = []

    for reg in REGIONS:
        name = reg["region_id"]
        bbox = reg["bbox"]
        days = reg["days"]
        label = reg["disturbance_class"]
        biome = reg["biome"]

        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{firms_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
        print(f"Fetching {name} ({label}, bbox={bbox})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
                if len(df) == 0:
                    print(f"  Warning: 0 detections for {name}")
                    continue

                # Subsample if necessary for class balance
                if len(df) > reg["max_samples"]:
                    df = df.sample(n=reg["max_samples"], random_state=42).reset_index(drop=True)

                df["region_id"] = name
                df["biome"] = biome
                df["disturbance_class"] = label
                all_dfs.append(df)
                print(f"  Successfully retrieved {len(df):,} satellite detections for {name}.")
        except Exception as e:
            safe_err = str(e).replace(firms_key, "***")
            print(f"  Error fetching {name}: {safe_err}")

    if not all_dfs:
        raise RuntimeError("Failed to retrieve any data from NASA FIRMS API.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Calculate physical satellite derived variables
    combined["delta_t"] = combined["bright_ti4"] - combined["bright_ti5"]
    combined["log_frp"] = np.log1p(np.maximum(0.0, combined["frp"]))
    combined["is_daytime"] = (combined["daynight"] == "D").astype(int)

    # Filter physical sanity bounds (remove cold background pixels or negative contrast artifacts)
    combined = combined[(combined["bright_ti4"] >= 250.0) & (combined["delta_t"] >= 0.0)].reset_index(drop=True)

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined):,} authentic NASA satellite detections across {len(REGIONS)} disturbance regimes to {OUTPUT_PATH}")
    print(combined["disturbance_class"].value_counts())
    return combined


if __name__ == "__main__":
    download_nasa_firms_data()
