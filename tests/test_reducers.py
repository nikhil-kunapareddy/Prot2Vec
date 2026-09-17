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


@pytest.fixture
def nonnegative():
    return np.abs(np.random.default_rng(0).standard_normal((30, 40)))


@pytest.fixture
def tiny():
    return np.random.default_rng(0).standard_normal((6, 4))


class TestLinearReducers:
    def test_svd_consumes_sparse_without_densifying(self, sparse):
        from prot2vec.reduction.reducers import TruncatedSVDReducer

        assert TruncatedSVDReducer().fit_transform(sparse).shape == (30, 2)

    def test_svd_is_deterministic(self, dense):
        from prot2vec.reduction.reducers import TruncatedSVDReducer

        np.testing.assert_allclose(
            TruncatedSVDReducer().fit_transform(dense),
            TruncatedSVDReducer().fit_transform(dense),
        )

    def test_svd_clamps_components_to_data(self):
        from prot2vec.reduction.reducers import TruncatedSVDReducer

        result = TruncatedSVDReducer(n_components=10).fit_transform(np.abs(np.eye(3)))
        assert result.shape[0] == 3

    def test_nmf_on_nonnegative_input(self, nonnegative):
        from prot2vec.reduction.reducers import NMFReducer

        assert NMFReducer().fit_transform(nonnegative).shape == (30, 2)

    def test_nmf_rejects_signed_input_with_guidance(self, dense):
        # Language model embeddings are signed; the message has to say what to
        # use instead rather than surfacing scikit-learn's generic complaint.
        from prot2vec.reduction.reducers import NMFReducer

        with pytest.raises(ValueError) as excinfo:
            NMFReducer().fit_transform(dense)
        message = str(excinfo.value)
        assert "non-negative" in message
        assert "pca" in message

    def test_random_projection_shape(self, dense):
        from prot2vec.reduction.reducers import RandomProjectionReducer

        assert RandomProjectionReducer().fit_transform(dense).shape == (30, 2)

    def test_random_projection_is_seeded(self, dense):
        from prot2vec.reduction.reducers import RandomProjectionReducer

        np.testing.assert_allclose(
            RandomProjectionReducer(random_state=1).fit_transform(dense),
            RandomProjectionReducer(random_state=1).fit_transform(dense),
        )
        assert not np.allclose(
            RandomProjectionReducer(random_state=1).fit_transform(dense),
            RandomProjectionReducer(random_state=2).fit_transform(dense),
        )


class TestManifoldReducers:
    @pytest.mark.parametrize(
        "name",
        ["KernelPCAReducer", "IsomapReducer", "MDSReducer", "SpectralReducer", "LLEReducer"],
    )
    def test_dense_shape(self, dense, name):
        import prot2vec.reduction.reducers as module

        assert getattr(module, name)().fit_transform(dense).shape == (30, 2)

    @pytest.mark.parametrize(
        "name",
        ["KernelPCAReducer", "IsomapReducer", "MDSReducer", "SpectralReducer", "LLEReducer"],
    )
    def test_sparse_input_accepted(self, sparse, name):
        import prot2vec.reduction.reducers as module

        assert getattr(module, name)().fit_transform(sparse).shape == (30, 2)

    @pytest.mark.parametrize(
        "name",
        ["KernelPCAReducer", "IsomapReducer", "MDSReducer", "SpectralReducer", "LLEReducer"],
    )
    def test_neighbourhoods_clamped_for_tiny_datasets(self, tiny, name):
        # Seed alignments are routinely smaller than the default n_neighbors,
        # which would otherwise raise inside scikit-learn.
        import prot2vec.reduction.reducers as module

        assert getattr(module, name)().fit_transform(tiny).shape == (6, 2)

    def test_kernel_appears_in_the_name(self):
        from prot2vec.reduction.reducers import KernelPCAReducer

        assert KernelPCAReducer().name == "kernel_pca"
        assert KernelPCAReducer(kernel="rbf").name == "kernel_pca_rbf"


class TestOptionalReducers:
    @pytest.mark.parametrize(
        ("name", "extra"), [("PHATEReducer", "phate"), ("PaCMAPReducer", "pacmap")]
    )
    def test_missing_backend_is_actionable(self, dense, name, extra):
        import prot2vec.reduction.reducers as module

        reducer = getattr(module, name)()
        with pytest.raises(ImportError) as excinfo:
            reducer.fit_transform(dense)
        message = str(excinfo.value)
        assert f"prot2vec[{extra}]" in message
        assert "pca" in message  # names a built-in alternative

    @pytest.mark.parametrize(
        ("name", "expected"), [("PHATEReducer", "phate"), ("PaCMAPReducer", "pacmap")]
    )
    def test_names_and_params(self, name, expected):
        import prot2vec.reduction.reducers as module

        reducer = getattr(module, name)()
        assert reducer.name == expected
        assert "n_components" in reducer.params


class TestReducerCatalogue:
    def test_every_reducer_reports_params(self):
        import prot2vec.reduction as module

        for name in module.__all__:
            if name == "DimReducer":
                continue
            reducer = getattr(module, name)()
            assert isinstance(reducer.params, dict)
            assert reducer.name

    def test_names_are_unique(self):
        import prot2vec.reduction as module

        names = [getattr(module, n)().name for n in module.__all__ if n != "DimReducer"]
        assert len(names) == len(set(names)) == 13
