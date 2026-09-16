"""Dimensionality reducers behind one interface.

The choice of reducer is not cosmetic. PCA is linear and deterministic, so it
shows what a representation encodes along its dominant axes of variance. UMAP
and t-SNE are non-linear and will happily invent visually crisp clusters from
noise, which is exactly why Prot2Vec reports ``trustworthiness`` alongside
every projection: the metric grades how much of the picture you should believe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .._matrix import DenseMatrix, EmbeddingMatrix, as_dense, is_sparse


class DimReducer(ABC):
    """Project a high-dimensional embedding down to a few components."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames, logs and result tables."""
        ...

    @abstractmethod
    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Reduce ``X`` to ``(n_samples, n_components)``.

        Parameters
        ----------
        X
            Embedding matrix, dense or sparse.

        Returns
        -------
        DenseMatrix
            Dense reduced coordinates.
        """
        ...

    @property
    def params(self) -> dict[str, Any]:
        """Effective settings, for run manifests and reproducibility.

        Recorded verbatim in ``run_manifest.json``: reproducing a figure means
        knowing the ``n_neighbors`` it was made with, not just that it was UMAP.
        """
        return {}

    def __repr__(self) -> str:
        """Render as ``ClassName(name='...')``."""
        return f"{type(self).__name__}(name={self.name!r})"


class PCAReducer(DimReducer):
    """Principal component analysis, preceded by feature standardisation.

    Deterministic and fast, and the only reducer here whose axes mean
    something: it is the right default when you want a projection you can
    reason about rather than the prettiest picture.

    Parameters
    ----------
    n_components
        Components to keep.
    random_state
        Seed for the randomised SVD solver.
    **kwargs
        Passed through to :class:`sklearn.decomposition.PCA`.
    """

    def __init__(self, n_components: int = 2, random_state: int = 0, **kwargs: Any) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``pca``."""
        return "pca"

    @property
    def params(self) -> dict[str, Any]:
        """Effective PCA settings."""
        return {
            "n_components": self._n_components,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Standardise ``X`` and project onto its leading principal components."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_dense = as_dense(X)
        n_components = min(self._n_components, *X_dense.shape)
        X_scaled = StandardScaler().fit_transform(X_dense)
        pca = PCA(n_components=n_components, random_state=self._random_state, **self._kwargs)
        return np.asarray(pca.fit_transform(X_scaled))


class UMAPReducer(DimReducer):
    """Uniform Manifold Approximation and Projection.

    The usual choice for protein embedding figures because it preserves local
    neighbourhoods well and accepts sparse input directly. Cosine distance is
    the default: embedding magnitude mostly tracks sequence length, which is
    not what family membership is about.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_neighbors
        Neighbourhood size. Small values emphasise local structure; with the
        few dozen sequences in a typical seed alignment, values above ~30 start
        to average families together.
    min_dist
        Minimum spacing between projected points.
    metric
        Distance metric.
    random_state
        Seed. Setting it makes UMAP single-threaded but reproducible, which is
        the right trade for a benchmark.
    **kwargs
        Passed through to :class:`umap.UMAP`.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._kwargs = dict(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Identifier: ``umap``."""
        return "umap"

    @property
    def params(self) -> dict[str, Any]:
        """Effective UMAP settings."""
        return dict(self._kwargs)

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Fit UMAP on ``X`` and return the projection.

        Raises
        ------
        ImportError
            If ``umap-learn`` cannot be imported, with the underlying error
            attached. ``umap.__init__`` eagerly imports ``parametric_umap``,
            which pulls in TensorFlow, which pulls in ``googleapiclient`` —
            so a clash anywhere in that chain surfaces here as an
            ``AttributeError`` rather than a missing module. Both cases are
            normalised into one actionable message, because "UMAP is installed
            but unusable" is far more common than "UMAP is absent".
        """
        try:
            import umap as umap_lib
        except Exception as exc:
            raise ImportError(
                "Could not import umap-learn, which UMAPReducer requires.\n"
                f"  Underlying error: {type(exc).__name__}: {exc}\n"
                "  If it is not installed:  pip install umap-learn\n"
                "  If it is installed, the import chain "
                "umap -> parametric_umap -> tensorflow is probably broken by a "
                "version clash. Either repair it (`pip install -U umap-learn "
                "numba pynndescent`, or `pip install -U pyparsing httplib2` for "
                "a googleapiclient clash), or use `reducer: {name: pca}` / "
                "`{name: tsne}` in your config to keep working now."
            ) from exc

        kwargs = dict(self._kwargs)
        n_samples = X.shape[0]
        # UMAP requires n_neighbors < n_samples; seed alignments are small.
        if kwargs["n_neighbors"] >= n_samples:
            kwargs["n_neighbors"] = max(2, n_samples - 1)
        return np.asarray(umap_lib.UMAP(**kwargs).fit_transform(X))


class TSNEReducer(DimReducer):
    """t-distributed stochastic neighbour embedding.

    Sparse input is compressed with truncated SVD first — t-SNE densifies its
    input, so an 8,000-column k-mer matrix would otherwise dominate runtime
    while adding mostly noise to the distance computation.

    Parameters
    ----------
    n_components
        Output dimensionality.
    perplexity
        Effective neighbourhood size. Must be less than the sample count;
        clamped automatically for small datasets.
    random_state
        Seed for reproducibility.
    n_svd_components
        Target width of the truncated-SVD pre-reduction for sparse input.
    **kwargs
        Passed through to :class:`sklearn.manifold.TSNE`.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 0,
        n_svd_components: int = 50,
        **kwargs: Any,
    ) -> None:
        self._n_svd = n_svd_components
        self._perplexity = perplexity
        self._random_state = random_state
        self._kwargs = dict(n_components=n_components, **kwargs)

    @property
    def name(self) -> str:
        """Identifier: ``tsne``."""
        return "tsne"

    @property
    def params(self) -> dict[str, Any]:
        """Effective t-SNE settings."""
        return {
            "perplexity": self._perplexity,
            "random_state": self._random_state,
            "n_svd_components": self._n_svd,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Optionally pre-reduce with SVD, then fit t-SNE."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.manifold import TSNE

        if is_sparse(X):
            n_svd = min(self._n_svd, X.shape[1] - 1, X.shape[0] - 1)
            if n_svd >= 2:
                X = TruncatedSVD(n_components=n_svd, random_state=self._random_state).fit_transform(
                    X
                )
            else:
                X = as_dense(X)

        # scikit-learn requires perplexity < n_samples.
        perplexity = min(self._perplexity, max(5.0, X.shape[0] - 1.0))
        tsne = TSNE(
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(tsne.fit_transform(X))
