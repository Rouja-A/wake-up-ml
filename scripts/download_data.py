#!/usr/bin/env python
"""Optional helper: download and extract UCI HAR into <out_dir>/UCI HAR Dataset.

    python scripts/download_data.py --out_dir data

Nothing else in the project depends on this script: you can also download the zip manually from
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones, extract it, and
pass the extracted folder via --data_dir or $UCI_HAR_DIR. The archive may contain a nested
"UCI HAR Dataset.zip"; this script handles that. Standard library only.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DatasetLayoutError, resolve_data_dir  # noqa: E402

URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Already present: {resolve_data_dir(out)}")
        return 0
    except DatasetLayoutError:
        pass

    zip_path = out / "uci_har_download.zip"
    for url in URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
    else:
        print("All download attempts failed. Download the zip manually (see --help) and extract it.")
        return 1

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    nested = next(out.glob("**/UCI HAR Dataset.zip"), None)
    if nested is not None:
        with zipfile.ZipFile(nested) as z:
            z.extractall(out)
    zip_path.unlink(missing_ok=True)
    try:
        print(f"OK: dataset at {resolve_data_dir(out)}")
        return 0
    except DatasetLayoutError as e:
        print(f"Extracted, but the layout is unexpected: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
