"""Tests for the dimensionality reducers."""

import numpy as np
import pytest
import scipy.sparse

from prot2vec.reduction.reducers import PCAReducer, TSNEReducer, UMAPReducer


@pytest.fixture
def dense():
    return np.random.default_rng(0).standard_normal((30, 40))


@pytest.fixture
def sparse():
    return scipy.sparse.random(30, 500, density=0.05, format="csr", random_state=0)


class TestPCAReducer:
    def test_dense_shape(self, dense):
        assert PCAReducer().fit_transform(dense).shape == (30, 2)

    def test_sparse_input_accepted(self, sparse):
        assert PCAReducer().fit_transform(sparse).shape == (30, 2)

    def test_deterministic(self, dense):
        np.testing.assert_allclose(
            PCAReducer().fit_transform(dense), PCAReducer().fit_transform(dense)
        )

    def test_components_clamped_to_data(self):
        # Fewer samples than requested components must not raise.
        tiny = np.random.default_rng(0).standard_normal((3, 2))
        assert PCAReducer(n_components=10).fit_transform(tiny).shape[0] == 3

    def test_name_and_params(self):
        reducer = PCAReducer(n_components=3, random_state=7)
        assert reducer.name == "pca"
        assert reducer.params["n_components"] == 3
        assert reducer.params["random_state"] == 7

    def test_repr(self):
        assert repr(PCAReducer()) == "PCAReducer(name='pca')"


class TestTSNEReducer:
    def test_dense_shape(self, dense):
        assert TSNEReducer(perplexity=10).fit_transform(dense).shape == (30, 2)

    def test_sparse_is_pre_reduced(self, sparse):
        assert TSNEReducer(perplexity=10).fit_transform(sparse).shape == (30, 2)

    def test_perplexity_clamped_for_tiny_input(self):
        tiny = np.random.default_rng(0).standard_normal((8, 4))
        assert TSNEReducer(perplexity=30).fit_transform(tiny).shape == (8, 2)

    def test_params_recorded(self):
        reducer = TSNEReducer(perplexity=12, random_state=3)
        assert reducer.params["perplexity"] == 12
        assert reducer.params["random_state"] == 3


class TestUMAPReducer:
    def test_name_and_params(self):
        reducer = UMAPReducer(n_neighbors=7, min_dist=0.2)
        assert reducer.name == "umap"
        assert reducer.params["n_neighbors"] == 7
        assert reducer.params["min_dist"] == 0.2

    def test_import_failure_is_actionable(self, dense, monkeypatch):
        # umap-learn eagerly imports TensorFlow via parametric_umap, so a
        # clash anywhere in that chain arrives as something other than
        # ImportError. All of it must surface as one actionable message.
        import builtins

        real_import = builtins.__import__

        def explode(name, *args, **kwargs):
            if name == "umap":
                raise AttributeError("module 'pyparsing' has no attribute 'DelimitedList'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", explode)
        with pytest.raises(ImportError) as excinfo:
            UMAPReducer().fit_transform(dense)
        message = str(excinfo.value)
        assert "umap-learn" in message
        assert "pca" in message  # points at a working alternative
        assert "DelimitedList" in message  # preserves the underlying cause
