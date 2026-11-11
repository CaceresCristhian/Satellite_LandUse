"""
Download and query official NASA MODIS MCD12Q1 Land Cover Type granules.

Uses NASA Earthdata Bearer Token securely from .env (EARTHDATA_TOKEN).
Queries the NASA Common Metadata Repository (CMR) for Collection C2484079608-LPCLOUD
(MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V061).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "01_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "modis_mcd12q1_granules.parquet"


def get_earthdata_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        return token.strip()

    # Search project-level .env, then parent workspace .env
    for candidate_env in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if candidate_env.exists():
            for line in candidate_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("EARTHDATA_TOKEN="):
                    return line.split("=", 1)[1].strip()

    raise RuntimeError("EARTHDATA_TOKEN not found in environment or .env file.")


# Official IGBP 17 Land Cover Classes in MCD12Q1
IGBP_CLASSES = {
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
}


def query_modis_granules(token: str, year: int = 2022, max_granules: int = 20) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    temporal = f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?collection_concept_id=C2484079608-LPCLOUD&temporal={temporal}&page_size={max_granules}"
    )

    print(f"Connecting to NASA CMR for MCD12Q1 Land Cover ({year})...")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    feed = r.json().get("feed", {})
    entries = feed.get("entry", [])
    print(f"Retrieved {len(entries)} MCD12Q1 granules from NASA LP DAAC cloud archive.")

    records = []
    for g in entries:
        title = g.get("title", "")
        size_mb = float(g.get("granule_size", 0.0))
        # Find download link to protected lpcloud bucket
        download_url = ""
        for link in g.get("links", []):
            href = link.get("href", "")
            if href.endswith(".hdf") or href.endswith(".tif"):
                download_url = href
                break

        # Parse MODIS horizontal/vertical tile identifiers (e.g. h17v10)
        parts = title.split(".")
        tile = parts[2] if len(parts) > 2 else "unknown"

        records.append({
            "granule_title": title,
            "tile": tile,
            "year": year,
            "size_mb": size_mb,
            "download_url": download_url,
            "concept_id": g.get("id", ""),
            "updated": g.get("updated", ""),
        })

    df = pd.DataFrame(records)
    return df


def main():
    token = get_earthdata_token()
    print("Authenticating with NASA Earthdata Token (loaded securely from environment)...")

    df = query_modis_granules(token, year=2022, max_granules=25)
    if not df.empty:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved {len(df)} MODIS MCD12Q1 granule records to {OUTPUT_PATH}")
        print("\nSample MODIS Land Cover Granules:")
        print(df[["granule_title", "tile", "size_mb", "download_url"]].head())


if __name__ == "__main__":
    main()
