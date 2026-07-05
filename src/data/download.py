"""Download the UCI Heart Disease dataset to data/raw/.

Usage: python -m src.data.download
"""

from pathlib import Path

from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("data/raw")
HEART_DISEASE_UCI_ID = 45


def download() -> Path:
    heart = fetch_ucirepo(id=HEART_DISEASE_UCI_ID)
    df = heart.data.features.copy()
    df["target"] = heart.data.targets
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "heart_disease.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return out


if __name__ == "__main__":
    download()
