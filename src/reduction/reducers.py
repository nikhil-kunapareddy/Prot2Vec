"""Dimensionality reduction wrappers with a shared interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DimReducer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


class PCAReducer(DimReducer):
    """PCA with automatic StandardScaler preprocessing."""

    def __init__(self, n_components: int = 2, random_state: int = 0, **kwargs: Any) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "pca"

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if hasattr(X, "toarray"):
            X = X.toarray()
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(
            n_components=self._n_components,
            random_state=self._random_state,
            **self._kwargs,
        )
        return pca.fit_transform(X_scaled)


class UMAPReducer(DimReducer):
    """UMAP with cosine metric by default (works well for both sparse and dense inputs)."""

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
        return "umap"

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import umap as umap_lib

        return umap_lib.UMAP(**self._kwargs).fit_transform(X)


class TSNEReducer(DimReducer):
    """t-SNE, preceded by TruncatedSVD when input is sparse (for speed)."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 0,
        n_svd_components: int = 50,
        **kwargs: Any,
    ) -> None:
        self._n_svd = n_svd_components
        self._kwargs = dict(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "tsne"

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import scipy.sparse
        from sklearn.decomposition import TruncatedSVD
        from sklearn.manifold import TSNE

        if scipy.sparse.issparse(X):
            n_svd = min(self._n_svd, X.shape[1] - 1, X.shape[0] - 1)
            X = TruncatedSVD(n_components=n_svd, random_state=0).fit_transform(X)

        return TSNE(**self._kwargs).fit_transform(X)
