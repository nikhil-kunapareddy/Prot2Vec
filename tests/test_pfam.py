"""Tests for the Pfam download and Stockholm parsing layer."""

import gzip

import pytest
import requests

from prot2vec.data.pfam import _looks_like_valid_gzip, download_pfam_seed, parse_pfam_families


def _stockholm_block(accession: str, identifier: str, sequences: dict[str, str]) -> str:
    lines = ["# STOCKHOLM 1.0", f"#=GF ID   {identifier}", f"#=GF AC   {accession}.25"]
    lines += [f"{name:<20} {seq}" for name, seq in sequences.items()]
    lines.append("//")
    return "\n".join(lines) + "\n"


@pytest.fixture
def seed_archive(tmp_path):
    """A miniature Pfam-A.seed.gz holding three families."""
    content = (
        _stockholm_block("PF00069", "Pkinase", {"KIN1_HUMAN/1-20": "ACDEFGHIKL-MNPQRSTVWY"})
        + _stockholm_block(
            "PF00072",
            "Response_reg",
            {
                "REG1_ECOLI/1-20": "MKTAYIAKQR.QISFVKSHFS",
                "REG2_ECOLI/1-20": "MKTAYIAKQRXQISFVKSHFT",
            },
        )
        + _stockholm_block("PF00001", "7tm_1", {"OPSD_BOVIN/1-20": "GFPINFLTLYVTVQHKKLRT"})
    )
    path = tmp_path / "Pfam-A.seed.35.0.gz"
    path.write_bytes(gzip.compress(content.encode()))
    return path


class TestParsePfamFamilies:
    def test_extracts_requested_families_only(self, seed_archive):
        records = parse_pfam_families(["PF00069", "PF00072"], seed_archive)
        assert set(records) == {"PF00069", "PF00072"}
        assert len(records["PF00069"]) == 1
        assert len(records["PF00072"]) == 2

    def test_version_suffix_stripped_from_accession(self, seed_archive):
        # The archive says "PF00069.25"; callers ask for "PF00069".
        assert parse_pfam_families(["PF00069"], seed_archive)["PF00069"]

    def test_sequence_content_preserved_including_gaps(self, seed_archive):
        record = parse_pfam_families(["PF00069"], seed_archive)["PF00069"][0]
        assert "-" in str(record.seq)  # cleaning happens in ProteinDataset, not here

    def test_absent_family_returns_empty_list(self, seed_archive, caplog):
        with caplog.at_level("WARNING"):
            records = parse_pfam_families(["PF99999"], seed_archive)
        assert records["PF99999"] == []
        assert "No sequences found" in caplog.text

    def test_missing_archive_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="prot2vec-download"):
            parse_pfam_families(["PF00069"], tmp_path / "absent.gz")

    def test_malformed_accession_rejected_before_reading(self, tmp_path):
        # Must fail without touching the (here nonexistent) 500 MB archive.
        with pytest.raises(ValueError, match="Malformed"):
            parse_pfam_families(["PF69"], tmp_path / "absent.gz")

    def test_duplicate_accessions_rejected(self, seed_archive):
        with pytest.raises(ValueError, match="Duplicate"):
            parse_pfam_families(["PF00069", "PF00069"], seed_archive)

    def test_stops_early_once_all_families_found(self, seed_archive):
        # PF00069 is the first block; parsing should not need the rest.
        assert len(parse_pfam_families(["PF00069"], seed_archive)["PF00069"]) == 1


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, content_length: bool = True):
        self._body = body
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))} if content_length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self  # type: ignore[assignment]
            raise error

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def payload():
    return gzip.compress(b"# STOCKHOLM 1.0\n#=GF AC   PF00069.25\nA/1-2 AC\n//\n" * 50)


class TestDownloadPfamSeed:
    def test_downloads_and_caches(self, tmp_path, payload, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
        path = download_pfam_seed(version="35.0", cache_dir=tmp_path)
        assert path.exists()
        assert path.name == "Pfam-A.seed.35.0.gz"

    def test_second_call_uses_the_cache(self, tmp_path, payload, monkeypatch):
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(1)
            return _FakeResponse(payload)

        monkeypatch.setattr(requests, "get", fake_get)
        download_pfam_seed(version="35.0", cache_dir=tmp_path)
        download_pfam_seed(version="35.0", cache_dir=tmp_path)
        assert len(calls) == 1

    def test_force_redownloads(self, tmp_path, payload, monkeypatch):
        calls = []
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: (calls.append(1), _FakeResponse(payload))[1]
        )
        download_pfam_seed(version="35.0", cache_dir=tmp_path)
        download_pfam_seed(version="35.0", cache_dir=tmp_path, force=True)
        assert len(calls) == 2

    def test_corrupt_cache_is_replaced(self, tmp_path, payload, monkeypatch):
        # A truncated earlier transfer must not be trusted forever.
        stale = tmp_path / "Pfam-A.seed.35.0.gz"
        stale.write_bytes(b"not gzip at all")
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
        path = download_pfam_seed(version="35.0", cache_dir=tmp_path)
        assert _looks_like_valid_gzip(path)

    def test_non_gzip_response_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(b"<html>404</html>"))
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            download_pfam_seed(version="35.0", cache_dir=tmp_path, max_retries=2)

    def test_no_partial_file_left_behind_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(b"garbage"))
        with pytest.raises(RuntimeError):
            download_pfam_seed(version="35.0", cache_dir=tmp_path, max_retries=1)
        assert list(tmp_path.glob("*.part")) == []

    def test_missing_release_is_not_retried(self, tmp_path, monkeypatch):
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(1)
            return _FakeResponse(b"", status=404)

        monkeypatch.setattr(requests, "get", fake_get)
        with pytest.raises(RuntimeError, match="not available"):
            download_pfam_seed(version="99.0", cache_dir=tmp_path, max_retries=3)
        assert len(calls) == 1  # a wrong release number will never succeed

    def test_server_error_is_retried(self, tmp_path, payload, monkeypatch):
        responses = [_FakeResponse(b"", status=503), _FakeResponse(payload)]
        monkeypatch.setattr(requests, "get", lambda *a, **k: responses.pop(0))
        monkeypatch.setattr("time.sleep", lambda _: None)
        assert download_pfam_seed(version="35.0", cache_dir=tmp_path, max_retries=3).exists()

    def test_progress_callback_invoked(self, tmp_path, payload, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
        seen = []
        download_pfam_seed(
            version="35.0", cache_dir=tmp_path, on_progress=lambda done, total: seen.append(done)
        )
        assert seen and seen[-1] == len(payload)
