"""Download and parse Pfam seed alignments from EBI."""
from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import requests
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

_PFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{version}/Pfam-A.seed.gz"


def download_pfam_seed(version: str = "35.0", cache_dir: str | Path = "data/raw") -> Path:
    """Download Pfam-A.seed.gz and cache it locally. Returns path to the cached file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"Pfam-A.seed.{version}.gz"

    if dest.exists():
        logger.info(f"Using cached Pfam seed: {dest}")
        return dest

    url = _PFAM_SEED_URL.format(version=version)
    logger.info(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            fh.write(chunk)
    logger.info(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def parse_pfam_families(
    pfam_ids: list[str],
    seed_path: str | Path,
) -> dict[str, list[SeqRecord]]:
    """Parse a Pfam-A.seed.gz file and extract records for the requested family IDs.

    Streams through the file without loading it fully into memory.
    Returns a dict mapping accession -> list of SeqRecord.
    """
    target = set(pfam_ids)
    records_by_family: dict[str, list[SeqRecord]] = {pid: [] for pid in pfam_ids}

    current_acc: str | None = None
    block_lines: list[str] = []

    def _flush(block: list[str], acc: str | None) -> None:
        if not block or acc not in target:
            return
        block_text = "# STOCKHOLM 1.0\n" + "".join(block) + "//\n"
        alignment = AlignIO.read(io.StringIO(block_text), "stockholm")
        records_by_family[acc].extend(alignment)

    with gzip.open(seed_path, "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#=GF AC"):
                _flush(block_lines, current_acc)
                block_lines = [line]
                current_acc = line.split()[2].split(".")[0]  # strip version suffix
            elif line.strip() == "//":
                block_lines.append(line)
                _flush(block_lines, current_acc)
                block_lines = []
                current_acc = None
            else:
                block_lines.append(line)

    for pid, recs in records_by_family.items():
        logger.info(f"{pid}: {len(recs)} sequences")

    return records_by_family
