"""k-mer TF-IDF: a bag-of-subsequences representation."""

from __future__ import annotations

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import SequenceEmbedder


class KmerEmbedder(SequenceEmbedder):
    """TF-IDF over overlapping residue k-mers.

    Treats each protein as a document of ``k``-length residue words. Unlike
    :class:`~prot2vec.embedders.composition.CompositionEmbedder` this keeps
    local order, which is enough to pick up short conserved motifs — the
    Walker A/B boxes of an ATPase, say — without any pretraining. TF-IDF
    weighting down-weights k-mers that occur everywhere, so the representation
    emphasises what is distinctive about a family.

    Dimensionality is ``20**k`` in the worst case (8,000 for the default
    ``k=3``), so the result is left as a sparse matrix; reducers and the
    pipeline handle sparse input directly.

    Parameters
    ----------
    k
        k-mer length. 2-4 is the useful range: ``k=1`` degenerates to
        composition, and above 4 the feature space is larger than any seed
        alignment can populate.
    min_df
        Ignore k-mers appearing in fewer than this many sequences (an integer
        count, or a fraction of the corpus). Trims the long tail of k-mers seen
        exactly once, which carry no comparative signal.
    """

    def __init__(self, k: int = 3, min_df: int | float = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k
        self.min_df = min_df
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
            min_df=min_df,
        )

    @property
    def name(self) -> str:
        """Identifier such as ``kmer_k3``."""
        return f"kmer_k{self.k}"

    @property
    def vocabulary_size(self) -> int:
        """Number of k-mers retained after fitting.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_transform`.
        """
        if not hasattr(self._vectorizer, "vocabulary_"):
            raise RuntimeError("vocabulary_size is only available after fit_transform().")
        return len(self._vectorizer.vocabulary_)

    def fit_transform(self, sequences: list[str]) -> spmatrix:
        """Fit the TF-IDF vocabulary on ``sequences`` and transform them.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse matrix of shape ``(len(sequences), n_kmers)``.

        Raises
        ------
        ValueError
            If ``sequences`` is empty, or if no sequence is long enough to
            contain a single k-mer.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")
        if all(len(s) < self.k for s in sequences):
            raise ValueError(
                f"No sequence is at least k={self.k} residues long, so no k-mer "
                "can be extracted. Lower k or check your input."
            )

        try:
            result: spmatrix = self._vectorizer.fit_transform(sequences)
        except ValueError as exc:
            raise ValueError(
                f"TF-IDF produced an empty vocabulary for k={self.k}, min_df={self.min_df}. "
                "Lower min_df or use more/longer sequences."
            ) from exc

        self._validate_rows(result, sequences)
        return result
