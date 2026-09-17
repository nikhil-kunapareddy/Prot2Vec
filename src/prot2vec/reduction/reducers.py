"""Dimensionality reducers behind one interface.

The choice of reducer is not cosmetic. PCA is linear and deterministic, so it
shows what a representation encodes along its dominant axes of variance. UMAP
and t-SNE are non-linear and will happily invent visually crisp clusters from
noise, which is exactly why Prot2Vec reports ``trustworthiness`` alongside
every projection: the metric grades how much of the picture you should believe.

Three families are available:

**Linear** — :class:`PCAReducer`, :class:`TruncatedSVDReducer`,
:class:`NMFReducer`, :class:`RandomProjectionReducer`. Deterministic, cheap,
and their axes mean something. ``svd`` is the one to reach for with sparse
k-mer input, since it is latent semantic analysis and needs no densification.
:class:`RandomProjectionReducer` is deliberately uninformed and belongs in a
benchmark as a control: whatever structure survives it was robust to begin
with.

**Manifold** — :class:`UMAPReducer`, :class:`TSNEReducer`,
:class:`IsomapReducer`, :class:`MDSReducer`, :class:`SpectralReducer`,
:class:`LLEReducer`, :class:`KernelPCAReducer`. These model curved structure
and produce the clearest figures, at the cost of stochasticity and
hyperparameter sensitivity. Run more than one before trusting a shape.

**Optional** — :class:`PHATEReducer` and :class:`PaCMAPReducer`, behind the
``[phate]`` and ``[pacmap]`` extras.

Because a run computes each embedding once and reuses it, comparing several
reducers on identical vectors costs almost nothing, and disagreement between
them is itself a result.
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


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class TruncatedSVDReducer(DimReducer):
    """Truncated SVD, i.e. latent semantic analysis.

    The right linear reducer for sparse input. Unlike :class:`PCAReducer` it
    never centres the data, so it consumes a k-mer TF-IDF matrix directly
    instead of densifying 8,000 columns first — the same reason LSA is used on
    term-document matrices.

    Parameters
    ----------
    n_components
        Components to keep.
    random_state
        Seed for the randomised solver.
    **kwargs
        Passed to :class:`sklearn.decomposition.TruncatedSVD`.
    """

    def __init__(self, n_components: int = 2, random_state: int = 0, **kwargs: Any) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``svd``."""
        return "svd"

    @property
    def params(self) -> dict[str, Any]:
        """Effective SVD settings."""
        return {
            "n_components": self._n_components,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Project onto the leading singular vectors, sparse input included."""
        from sklearn.decomposition import TruncatedSVD

        # TruncatedSVD needs strictly fewer components than features.
        n_components = min(self._n_components, min(X.shape) - 1)
        if n_components < 1:
            return as_dense(X)[:, : self._n_components]
        svd = TruncatedSVD(
            n_components=n_components, random_state=self._random_state, **self._kwargs
        )
        return np.asarray(svd.fit_transform(X))


class NMFReducer(DimReducer):
    """Non-negative matrix factorisation.

    Constrains both factors to be non-negative, so components add rather than
    cancel. On k-mer counts that yields parts-based components — groups of
    k-mers that co-occur — which are easier to read as motifs than the signed
    mixtures PCA returns.

    Requires non-negative input, which rules out language-model embeddings.

    Parameters
    ----------
    n_components
        Components to learn.
    random_state
        Seed for the initialisation.
    max_iter
        Coordinate-descent iterations.
    **kwargs
        Passed to :class:`sklearn.decomposition.NMF`.
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 0,
        max_iter: int = 500,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._max_iter = max_iter
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``nmf``."""
        return "nmf"

    @property
    def params(self) -> dict[str, Any]:
        """Effective NMF settings."""
        return {
            "n_components": self._n_components,
            "random_state": self._random_state,
            "max_iter": self._max_iter,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Factorise ``X`` into non-negative components.

        Raises
        ------
        ValueError
            If ``X`` contains negative values, naming the likely cause rather
            than surfacing scikit-learn's generic message.
        """
        from sklearn.decomposition import NMF

        minimum = X.min() if is_sparse(X) else float(np.min(as_dense(X)))
        if float(minimum) < 0.0:
            raise ValueError(
                "NMF requires non-negative input, but this embedding contains "
                f"values as low as {float(minimum):.3f}. Protein language model "
                "embeddings are signed — use pca, svd or umap for those, and "
                "keep nmf for counts such as composition, dipeptide, k-mer or "
                "one-hot."
            )

        nmf = NMF(
            n_components=min(self._n_components, min(X.shape)),
            random_state=self._random_state,
            max_iter=self._max_iter,
            **self._kwargs,
        )
        return np.asarray(nmf.fit_transform(X))


class RandomProjectionReducer(DimReducer):
    """Gaussian random projection — the control condition.

    Projects onto random directions, using no property of the data at all.
    Johnson-Lindenstrauss guarantees that pairwise distances are approximately
    preserved, so this is not noise: it is the amount of structure visible
    *without* any fitting. Reporting it alongside UMAP is the cleanest way to
    show that a clustered-looking projection reflects the representation rather
    than the reducer's appetite for finding clusters.

    Parameters
    ----------
    n_components
        Output dimensionality.
    random_state
        Seed for the projection matrix.
    **kwargs
        Passed to :class:`sklearn.random_projection.GaussianRandomProjection`.
    """

    def __init__(self, n_components: int = 2, random_state: int = 0, **kwargs: Any) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``random_projection``."""
        return "random_projection"

    @property
    def params(self) -> dict[str, Any]:
        """Effective projection settings."""
        return {
            "n_components": self._n_components,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Project ``X`` onto random Gaussian directions."""
        from sklearn.random_projection import GaussianRandomProjection

        projector = GaussianRandomProjection(
            n_components=min(self._n_components, X.shape[1]),
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(projector.fit_transform(X))


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------


def _clamp_neighbors(requested: int, n_samples: int, minimum: int = 2) -> int:
    """Clamp a neighbourhood size to what ``n_samples`` can support.

    Pfam seed alignments routinely hold fewer sequences than the default
    ``n_neighbors`` of these estimators, which would otherwise raise deep
    inside scikit-learn on a perfectly reasonable dataset.
    """
    return max(minimum, min(requested, n_samples - 1))


class KernelPCAReducer(DimReducer):
    """PCA in an implicit feature space.

    Keeps PCA's determinism while allowing non-linear structure, which puts it
    between PCA and UMAP: more expressive than a linear projection, far less
    prone to inventing clusters than a neighbour-graph method. The ``cosine``
    kernel is the default because embedding magnitude largely tracks sequence
    length, which is not what family membership is about.

    Parameters
    ----------
    n_components
        Components to keep.
    kernel
        ``"cosine"``, ``"rbf"``, ``"poly"``, ``"sigmoid"`` or ``"linear"``.
    gamma
        Kernel width for ``rbf``/``poly``/``sigmoid``. ``None`` uses
        scikit-learn's ``1 / n_features`` default.
    random_state
        Seed for the arpack solver.
    **kwargs
        Passed to :class:`sklearn.decomposition.KernelPCA`.
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "cosine",
        gamma: float | None = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._kernel = kernel
        self._gamma = gamma
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier such as ``kernel_pca`` or ``kernel_pca_rbf``."""
        suffix = "" if self._kernel == "cosine" else f"_{self._kernel}"
        return f"kernel_pca{suffix}"

    @property
    def params(self) -> dict[str, Any]:
        """Effective kernel PCA settings."""
        return {
            "n_components": self._n_components,
            "kernel": self._kernel,
            "gamma": self._gamma,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Project ``X`` onto kernel principal components."""
        from sklearn.decomposition import KernelPCA

        X_dense = as_dense(X)
        kpca = KernelPCA(
            n_components=min(self._n_components, *X_dense.shape),
            kernel=self._kernel,
            gamma=self._gamma,
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(kpca.fit_transform(X_dense))


class IsomapReducer(DimReducer):
    """Isometric mapping: MDS on geodesic distances.

    Builds a neighbour graph and preserves shortest-path distances through it,
    so unlike t-SNE and UMAP it tries to keep *global* structure — the relative
    distances between families, not just the tightness of each one. Worth
    running when the question is how families relate to each other rather than
    whether they separate.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_neighbors
        Neighbourhood size for the graph; clamped to the sample count.
    **kwargs
        Passed to :class:`sklearn.manifold.Isomap`.
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 10, **kwargs: Any) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``isomap``."""
        return "isomap"

    @property
    def params(self) -> dict[str, Any]:
        """Effective Isomap settings."""
        return {
            "n_components": self._n_components,
            "n_neighbors": self._n_neighbors,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` by preserving geodesic distances."""
        from sklearn.manifold import Isomap

        X_dense = as_dense(X)
        isomap = Isomap(
            n_components=min(self._n_components, X_dense.shape[1]),
            n_neighbors=_clamp_neighbors(self._n_neighbors, X_dense.shape[0]),
            **self._kwargs,
        )
        return np.asarray(isomap.fit_transform(X_dense))


class MDSReducer(DimReducer):
    """Multidimensional scaling.

    Places points so that their 2-D distances match their original distances as
    closely as possible, with no neighbour graph and no perplexity to tune.
    That makes it the most literal possible picture of the distance matrix, and
    a useful reference when a neighbour-based method produces something
    surprising.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_init
        Restarts with different initialisations; the best stress wins.
    random_state
        Seed for the initialisations.
    **kwargs
        Passed to :class:`sklearn.manifold.MDS`.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_init: int = 4,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._n_init = n_init
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``mds``."""
        return "mds"

    @property
    def params(self) -> dict[str, Any]:
        """Effective MDS settings."""
        return {
            "n_components": self._n_components,
            "n_init": self._n_init,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` so that pairwise distances are preserved."""
        import inspect

        from sklearn.manifold import MDS

        X_dense = as_dense(X)
        kwargs = dict(self._kwargs)
        # scikit-learn is changing MDS's `init` default from "random" to
        # "classical_mds", which would silently move every MDS number in a
        # published benchmark on upgrade. Pin it where the parameter exists.
        if "init" not in kwargs and "init" in inspect.signature(MDS).parameters:
            kwargs["init"] = "random"
        mds = MDS(
            n_components=self._n_components,
            n_init=self._n_init,
            random_state=self._random_state,
            **kwargs,
        )
        return np.asarray(mds.fit_transform(X_dense))


class SpectralReducer(DimReducer):
    """Laplacian eigenmaps.

    Takes the leading eigenvectors of the neighbour-graph Laplacian, which is
    the same mathematics that underlies spectral clustering. It therefore tends
    to separate groups that are connected internally and sparsely linked to
    each other — exactly the structure protein families have when homology is
    transitive within a family but not across.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_neighbors
        Neighbourhood size for the affinity graph; clamped to the sample count.
    random_state
        Seed for the eigensolver.
    **kwargs
        Passed to :class:`sklearn.manifold.SpectralEmbedding`.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``spectral``."""
        return "spectral"

    @property
    def params(self) -> dict[str, Any]:
        """Effective spectral embedding settings."""
        return {
            "n_components": self._n_components,
            "n_neighbors": self._n_neighbors,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` using the graph Laplacian's leading eigenvectors."""
        from sklearn.manifold import SpectralEmbedding

        X_dense = as_dense(X)
        spectral = SpectralEmbedding(
            n_components=min(self._n_components, X_dense.shape[0] - 1),
            n_neighbors=_clamp_neighbors(self._n_neighbors, X_dense.shape[0]),
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(spectral.fit_transform(X_dense))


class LLEReducer(DimReducer):
    """Locally linear embedding.

    Represents every point as a weighted combination of its neighbours and
    finds the low-dimensional coordinates that keep those weights. Cheaper than
    Isomap because it never computes all-pairs shortest paths, and unlike UMAP
    it has no stochastic optimisation stage.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_neighbors
        Neighbourhood size; clamped to exceed ``n_components`` and stay below
        the sample count, both of which LLE requires.
    random_state
        Seed for the eigensolver.
    **kwargs
        Passed to :class:`sklearn.manifold.LocallyLinearEmbedding`.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``lle``."""
        return "lle"

    @property
    def params(self) -> dict[str, Any]:
        """Effective LLE settings."""
        return {
            "n_components": self._n_components,
            "n_neighbors": self._n_neighbors,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` by preserving local reconstruction weights."""
        from sklearn.manifold import LocallyLinearEmbedding

        X_dense = as_dense(X)
        n_components = min(self._n_components, X_dense.shape[1])
        # LLE needs n_neighbors > n_components for the weights to be determined.
        n_neighbors = max(n_components + 1, _clamp_neighbors(self._n_neighbors, X_dense.shape[0]))
        lle = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=min(n_neighbors, X_dense.shape[0] - 1),
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(lle.fit_transform(X_dense))


# ---------------------------------------------------------------------------
# Optional third-party
# ---------------------------------------------------------------------------


def _import_optional(module: str, extra: str, reducer: str) -> Any:
    """Import an optional reducer backend with an actionable error.

    Raises
    ------
    ImportError
        If the module cannot be imported, quoting the underlying error. These
        packages pull in large numerical stacks, so a broken install is at
        least as likely as a missing one.
    """
    try:
        return __import__(module)
    except Exception as exc:
        raise ImportError(
            f"Could not import {module}, which {reducer} requires.\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            f'  Install it with: pip install "prot2vec[{extra}]"\n'
            "  Or choose a built-in reducer: pca, svd, umap, tsne, isomap, "
            "mds, spectral, lle, kernel_pca, nmf, random_projection."
        ) from exc


class PHATEReducer(DimReducer):
    """PHATE: potential of heat diffusion for affinity-based transition embedding.

    Designed for data with continuous trajectories rather than discrete
    clusters, which is why it became standard in single-cell work. That makes
    it the right choice when the biology is a gradient — a protein family with
    progressive divergence, or an engineered mutational series — where UMAP
    tends to shatter a continuum into arbitrary islands.

    Requires the ``[phate]`` extra.

    Parameters
    ----------
    n_components
        Output dimensionality.
    knn
        Neighbours used to build the affinity graph.
    decay
        Alpha-decay kernel exponent.
    random_state
        Seed.
    **kwargs
        Passed to :class:`phate.PHATE`.
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._knn = knn
        self._decay = decay
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``phate``."""
        return "phate"

    @property
    def params(self) -> dict[str, Any]:
        """Effective PHATE settings."""
        return {
            "n_components": self._n_components,
            "knn": self._knn,
            "decay": self._decay,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` with PHATE.

        Raises
        ------
        ImportError
            If the ``[phate]`` extra is not installed.
        """
        phate = _import_optional("phate", "phate", "PHATEReducer")

        X_dense = as_dense(X)
        operator = phate.PHATE(
            n_components=self._n_components,
            knn=_clamp_neighbors(self._knn, X_dense.shape[0]),
            decay=self._decay,
            random_state=self._random_state,
            verbose=0,
            **self._kwargs,
        )
        return np.asarray(operator.fit_transform(X_dense))


class PaCMAPReducer(DimReducer):
    """PaCMAP: pairwise controlled manifold approximation.

    Balances near, mid-range and far pairs explicitly, which is its reason for
    existing: UMAP and t-SNE optimise local neighbourhoods and leave the
    distance *between* clusters essentially arbitrary. When you want to read
    "these two families are closer to each other than to the third" off a
    figure, this preserves global geometry considerably better.

    Requires the ``[pacmap]`` extra.

    Parameters
    ----------
    n_components
        Output dimensionality.
    n_neighbors
        Neighbours per point.
    mn_ratio
        Mid-near pairs as a fraction of ``n_neighbors``.
    fp_ratio
        Further pairs as a fraction of ``n_neighbors``.
    random_state
        Seed.
    **kwargs
        Passed to :class:`pacmap.PaCMAP`.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._mn_ratio = mn_ratio
        self._fp_ratio = fp_ratio
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Identifier: ``pacmap``."""
        return "pacmap"

    @property
    def params(self) -> dict[str, Any]:
        """Effective PaCMAP settings."""
        return {
            "n_components": self._n_components,
            "n_neighbors": self._n_neighbors,
            "MN_ratio": self._mn_ratio,
            "FP_ratio": self._fp_ratio,
            "random_state": self._random_state,
            **self._kwargs,
        }

    def fit_transform(self, X: EmbeddingMatrix) -> DenseMatrix:
        """Embed ``X`` with PaCMAP.

        Raises
        ------
        ImportError
            If the ``[pacmap]`` extra is not installed.
        """
        pacmap = _import_optional("pacmap", "pacmap", "PaCMAPReducer")

        X_dense = as_dense(X)
        operator = pacmap.PaCMAP(
            n_components=self._n_components,
            n_neighbors=_clamp_neighbors(self._n_neighbors, X_dense.shape[0]),
            MN_ratio=self._mn_ratio,
            FP_ratio=self._fp_ratio,
            random_state=self._random_state,
            **self._kwargs,
        )
        return np.asarray(operator.fit_transform(X_dense))
