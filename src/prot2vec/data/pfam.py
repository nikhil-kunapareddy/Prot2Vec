"""Download and parse Pfam seed alignments from the EBI FTP mirror.

The seed alignment is the manually curated subset of each family, which is what
makes it a fair benchmark target: every sequence in a family is there because a
curator put it there, not because a profile HMM matched it.

``Pfam-A.seed.gz`` is roughly 500 MB, so the download is cached, written
atomically, and verified before it is trusted — a half-finished transfer that
looked like a valid cache used to poison every later run.
"""

from __future__ import annotations

import gzip
import io
import logging
import time
from collections.abc import Callable
from pathlib import Path

import requests
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord

from .families import validate_accessions

logger = logging.getLogger(__name__)

_PFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{version}/Pfam-A.seed.gz"
_CHUNK_BYTES = 1 << 20  # 1 MiB
_DOWNLOAD_TIMEOUT = (30, 300)  # (connect, read) seconds


def _looks_like_valid_gzip(path: Path) -> bool:
    """Return ``True`` if ``path`` can be opened and read as gzip text."""
    try:
        with gzip.open(path, "rt", encoding="latin-1") as fh:
            return bool(fh.read(2048))
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def download_pfam_seed(
    version: str = "35.0",
    cache_dir: str | Path = "data/raw",
    max_retries: int = 3,
    force: bool = False,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download ``Pfam-A.seed.gz`` for ``version`` and cache it locally.

    Parameters
    ----------
    version
        Pfam release, e.g. ``"35.0"``. Pinning a release keeps benchmark
        numbers comparable across machines and over time.
    cache_dir
        Directory for the cached archive. Created if missing.
    max_retries
        Attempts before giving up, with linear back-off between them.
    force
        Re-download even when a valid cached copy exists.
    on_progress
        Called as ``on_progress(bytes_done, total_bytes_or_None)`` during the
        transfer, for CLI progress reporting.

    Returns
    -------
    pathlib.Path
        Path to the cached ``.gz`` archive.

    Raises
    ------
    requests.HTTPError
        If the release does not exist on the mirror.
    RuntimeError
        If every attempt fails, or the transfer completes but is not readable
        gzip.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"Pfam-A.seed.{version}.gz"

    if dest.exists() and not force:
        if _looks_like_valid_gzip(dest):
            logger.info("Using cached Pfam seed: %s", dest)
            return dest
        logger.warning("Cached file %s is corrupt or truncated — re-downloading.", dest)
        dest.unlink()

    url = _PFAM_SEED_URL.format(version=version)
    # Download beside the target, then rename: an interrupted transfer leaves a
    # .part file that is never mistaken for a complete cache.
    partial = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt, max_retries)
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or 0) or None

                done = 0
                with partial.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                        fh.write(chunk)
                        done += len(chunk)
                        if on_progress is not None:
                            on_progress(done, total)

            if total is not None and done < total:
                raise OSError(f"Truncated download: got {done} of {total} bytes")
            if not _looks_like_valid_gzip(partial):
                raise OSError("Downloaded file is not readable gzip")

            partial.replace(dest)
            logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
            return dest

        except requests.HTTPError as exc:
            partial.unlink(missing_ok=True)
            status = exc.response.status_code if exc.response is not None else None
            # 4xx other than 429 will never succeed on retry (wrong release,
            # moved mirror); 5xx and rate limits are worth another attempt.
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(
                    f"Pfam {version} is not available at {url} (HTTP {status}). "
                    "Check the release number against "
                    "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/"
                ) from exc
            last_error = exc
            logger.warning("Download attempt %d failed: HTTP %s", attempt, status)
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(2.0 * attempt)

    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts: {last_error}"
    ) from last_error


def parse_pfam_families(
    pfam_ids: list[str],
    seed_path: str | Path,
) -> dict[str, list[SeqRecord]]:
    """Extract the seed alignments of specific families from a Pfam release.

    The archive holds every family in one Stockholm stream, so this makes a
    single streaming pass and keeps only the requested blocks — the whole file
    never has to fit in memory.

    Parameters
    ----------
    pfam_ids
        Accessions to extract, e.g. ``["PF00069", "PF00072"]``. Validated
        before the file is opened.
    seed_path
        Path to a ``Pfam-A.seed.gz`` archive, as returned by
        :func:`download_pfam_seed`.

    Returns
    -------
    dict of str to list of Bio.SeqRecord.SeqRecord
        Accession to its seed records. Families that the release does not
        contain map to an empty list.

    Raises
    ------
    FileNotFoundError
        If ``seed_path`` does not exist.
    ValueError
        If any requested accession is malformed or duplicated.
    """
    validate_accessions(pfam_ids)

    seed_path = Path(seed_path)
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Pfam seed archive not found: {seed_path}. "
            "Run `prot2vec-download` first, or pass a different --cache-dir."
        )

    target = set(pfam_ids)
    records_by_family: dict[str, list[SeqRecord]] = {pid: [] for pid in pfam_ids}
    remaining = set(target)

    current_acc: str | None = None
    block_lines: list[str] = []

    def _flush(block: list[str], acc: str | None) -> None:
        if not block or acc not in target:
            return
        block_text = "# STOCKHOLM 1.0\n" + "".join(block) + "//\n"
        try:
            alignment = AlignIO.read(io.StringIO(block_text), "stockholm")
        except ValueError as exc:  # malformed block in the release
            logger.warning("Skipping unparseable block for %s: %s", acc, exc)
            return
        records_by_family[acc].extend(alignment)
        remaining.discard(acc)

    with gzip.open(seed_path, "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#=GF AC"):
                _flush(block_lines, current_acc)
                block_lines = [line]
                current_acc = line.split()[2].split(".")[0]  # strip the version suffix
            elif line.strip() == "//":
                block_lines.append(line)
                _flush(block_lines, current_acc)
                block_lines = []
                current_acc = None
                if not remaining:
                    break  # every requested family found; stop reading
            else:
                block_lines.append(line)
        else:
            _flush(block_lines, current_acc)

    for pid, records in records_by_family.items():
        logger.info("%s: %d seed sequences", pid, len(records))

    missing = [pid for pid, records in records_by_family.items() if not records]
    if missing:
        logger.warning(
            "No sequences found for %s — check the accession(s) exist in Pfam %s.",
            ", ".join(missing),
            seed_path.name,
        )

    return records_by_family
