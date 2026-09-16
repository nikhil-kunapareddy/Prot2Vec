"""A small curated catalogue of well-characterised Pfam families.

Benchmarks are only interpretable if you know what the families *are*: a run
that separates two unrelated folds is a much weaker result than one that
separates two families from the same superfamily. This module keeps the
accession -> human-readable identity mapping in one place so that configs,
CLI output and figures all agree, and so that a typo in an accession is caught
before a 500 MB download rather than after it.

The catalogue is deliberately partial. Any valid Pfam accession works
throughout Prot2Vec; entries here only add a display name and a description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Pfam accessions are the letters ``PF`` followed by exactly five digits.
PFAM_ACCESSION_RE = re.compile(r"^PF\d{5}$")


@dataclass(frozen=True)
class PfamFamily:
    """Identity of a single Pfam family.

    Parameters
    ----------
    accession
        Stable Pfam accession, e.g. ``"PF00069"``.
    identifier
        Short Pfam ID used in the literature, e.g. ``"Pkinase"``.
    description
        One-line description of the domain.
    """

    accession: str
    identifier: str
    description: str

    def __str__(self) -> str:
        """Render as ``PF00069 (Pkinase)``."""
        return f"{self.accession} ({self.identifier})"


#: Curated families, chosen to span easy and hard separation problems.
CURATED_FAMILIES: dict[str, PfamFamily] = {
    f.accession: f
    for f in (
        PfamFamily("PF00001", "7tm_1", "7 transmembrane receptor (rhodopsin family GPCR)"),
        PfamFamily("PF00002", "7tm_2", "7 transmembrane receptor (secretin family GPCR)"),
        PfamFamily("PF00004", "AAA", "ATPase family associated with various cellular activities"),
        PfamFamily("PF00005", "ABC_tran", "ABC transporter nucleotide-binding domain"),
        PfamFamily("PF00006", "ATP-synt_ab", "ATP synthase alpha/beta family, nucleotide-binding"),
        PfamFamily("PF00027", "cNMP_binding", "Cyclic nucleotide-binding domain"),
        PfamFamily("PF00041", "fn3", "Fibronectin type III domain"),
        PfamFamily("PF00043", "GST_C", "Glutathione S-transferase, C-terminal domain"),
        PfamFamily("PF00069", "Pkinase", "Protein kinase domain (serine/threonine and tyrosine)"),
        PfamFamily("PF00072", "Response_reg", "Response regulator receiver domain (two-component)"),
        PfamFamily("PF00076", "RRM_1", "RNA recognition motif (a.k.a. RBD or RNP domain)"),
        PfamFamily("PF00089", "Trypsin", "Trypsin-like serine protease"),
        PfamFamily("PF00096", "zf-C2H2", "Zinc finger, C2H2 type"),
        PfamFamily("PF00104", "Hormone_recep", "Nuclear hormone receptor ligand-binding domain"),
        PfamFamily("PF00106", "adh_short", "Short-chain dehydrogenase/reductase"),
        PfamFamily("PF00112", "Peptidase_C1", "Papain family cysteine protease"),
        PfamFamily("PF00127", "Copper-bind", "Copper-binding proteins, plastocyanin/azurin family"),
        PfamFamily(
            "PF00144", "Beta-lactamase", "Beta-lactamase / penicillin-binding transpeptidase"
        ),
        PfamFamily("PF00171", "Aldedh", "Aldehyde dehydrogenase family"),
        PfamFamily("PF00186", "DHFR_1", "Dihydrofolate reductase"),
        PfamFamily("PF00201", "UDPGT", "UDP-glucuronosyl / UDP-glucosyl transferase"),
        PfamFamily("PF00483", "NTP_transferase", "Nucleotidyl transferase"),
        PfamFamily(
            "PF00528", "BPD_transp_1", "Binding-protein-dependent transport, inner membrane"
        ),
        PfamFamily("PF00650", "CRAL_TRIO", "CRAL/TRIO lipid-binding domain"),
        PfamFamily("PF07714", "PK_Tyr_Ser-Thr", "Protein tyrosine and serine/threonine kinase"),
    )
}


def describe(accession: str) -> str:
    """Return a human-readable label for ``accession``.

    Falls back to the bare accession when it is not in the curated catalogue,
    so unknown-but-valid families still render sensibly.

    Examples
    --------
    >>> describe("PF00069")
    'PF00069 (Pkinase)'
    >>> describe("PF99999")
    'PF99999'
    """
    family = CURATED_FAMILIES.get(accession)
    return str(family) if family else accession


def validate_accessions(accessions: list[str]) -> None:
    """Raise :class:`ValueError` if any accession is not well-formed.

    Catching this up front matters because parsing a Pfam seed release is a
    single streaming pass over a ~500 MB archive: a typo would otherwise cost
    minutes and return an empty family.

    Raises
    ------
    ValueError
        If ``accessions`` is empty or contains a malformed accession.
    """
    if not accessions:
        raise ValueError("No Pfam families requested — specify at least one accession.")

    malformed = [a for a in accessions if not PFAM_ACCESSION_RE.match(str(a))]
    if malformed:
        raise ValueError(
            f"Malformed Pfam accession(s): {malformed}. "
            "Expected the letters 'PF' followed by five digits, e.g. 'PF00069'."
        )

    duplicates = sorted({a for a in accessions if accessions.count(a) > 1})
    if duplicates:
        raise ValueError(f"Duplicate Pfam accession(s) requested: {duplicates}")
