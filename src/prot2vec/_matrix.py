"""The embedding-matrix abstraction: its type and its two conversions.

Prot2Vec passes embeddings around as either dense arrays or scipy sparse
matrices — a k-mer representation is ``20**k`` columns wide and densifying it
early would be wasteful. Naming that union once keeps signatures honest, and
keeping the conversions here means there is one definition of "make this
dense" rather than a copy in every module that needs one.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from scipy.sparse import csr_matrix, spmatrix

#: An embedding matrix: dense ``(n_samples, n_features)`` or scipy sparse.
EmbeddingMatrix: TypeAlias = "np.ndarray | spmatrix"

#: A dense coordinate array, as produced by every reducer.
DenseMatrix: TypeAlias = "np.ndarray"


def is_sparse(X: EmbeddingMatrix) -> bool:
    """Return ``True`` if ``X`` is a scipy sparse matrix."""
    return hasattr(X, "toarray")


def as_dense(X: EmbeddingMatrix) -> DenseMatrix:
    """Densify a sparse matrix, passing dense arrays straight through.

    Several metrics (trustworthiness, silhouette, k-means) require dense input.
    Converting at the point of use rather than at the point of creation keeps
    the memory cost to the one place that cannot avoid it.
    """
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray())
    return np.asarray(X)


def as_csr(X: EmbeddingMatrix) -> csr_matrix:
    """Return ``X`` as a CSR matrix, for a canonical on-disk sparse format.

    Raises
    ------
    TypeError
        If ``X`` is not sparse.
    """
    if not hasattr(X, "tocsr"):
        raise TypeError(f"Expected a sparse matrix, got {type(X).__name__}.")
    return X.tocsr()


__all__ = ["DenseMatrix", "EmbeddingMatrix", "as_csr", "as_dense", "is_sparse"]
