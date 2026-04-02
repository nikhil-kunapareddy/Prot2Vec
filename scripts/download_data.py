"""CLI script: download and cache Pfam seed alignment."""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Pfam-A.seed.gz from EBI and cache locally."
    )
    parser.add_argument("--version", default="35.0", help="Pfam release version (default: 35.0)")
    parser.add_argument(
        "--cache-dir", default="data/raw", help="Directory to save the file (default: data/raw)"
    )
    args = parser.parse_args()

    from prot2vec.data.pfam import download_pfam_seed

    path = download_pfam_seed(version=args.version, cache_dir=args.cache_dir)
    print(f"Pfam seed available at: {path}")


if __name__ == "__main__":
    main()
