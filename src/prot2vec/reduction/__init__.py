"""Dimensionality reducers. All implement :class:`DimReducer`.

Linear and deterministic: ``pca``, ``svd``, ``nmf``, ``random_projection``.
Manifold: ``umap``, ``tsne``, ``isomap``, ``mds``, ``spectral``, ``lle``,
``kernel_pca``. Optional extras: ``phate``, ``pacmap``.
"""

from .reducers import (
    DimReducer,
    IsomapReducer,
    KernelPCAReducer,
    LLEReducer,
    MDSReducer,
    NMFReducer,
    PaCMAPReducer,
    PCAReducer,
    PHATEReducer,
    RandomProjectionReducer,
    SpectralReducer,
    TruncatedSVDReducer,
    TSNEReducer,
    UMAPReducer,
)

__all__ = [
    "DimReducer",
    "IsomapReducer",
    "KernelPCAReducer",
    "LLEReducer",
    "MDSReducer",
    "NMFReducer",
    "PCAReducer",
    "PHATEReducer",
    "PaCMAPReducer",
    "RandomProjectionReducer",
    "SpectralReducer",
    "TSNEReducer",
    "TruncatedSVDReducer",
    "UMAPReducer",
]
