"""Tests for embedders."""
import numpy as np

from prot2vec.embedders.composition import CompositionEmbedder
from prot2vec.embedders.kmer import KmerEmbedder

SEQS = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ", "ACACACACACACACACACAC"]


class TestCompositionEmbedder:
    def test_output_shape(self):
        emb = CompositionEmbedder()
        X = emb.fit_transform(SEQS)
        assert X.shape == (len(SEQS), 20)

    def test_rows_sum_to_one(self):
        emb = CompositionEmbedder()
        X = emb.fit_transform(SEQS)
        np.testing.assert_allclose(X.sum(axis=1), np.ones(len(SEQS)), atol=1e-5)

    def test_name(self):
        assert CompositionEmbedder().name == "composition"

    def test_identical_sequences_identical_vectors(self):
        emb = CompositionEmbedder()
        X = emb.fit_transform([SEQS[0], SEQS[0]])
        np.testing.assert_array_equal(X[0], X[1])


class TestKmerEmbedder:
    def test_output_shape(self):
        emb = KmerEmbedder(k=3)
        X = emb.fit_transform(SEQS)
        assert X.shape[0] == len(SEQS)

    def test_name(self):
        assert KmerEmbedder(k=3).name == "kmer_k3"
        assert KmerEmbedder(k=2).name == "kmer_k2"

    def test_sparse_output(self):
        import scipy.sparse
        emb = KmerEmbedder(k=3)
        X = emb.fit_transform(SEQS)
        assert scipy.sparse.issparse(X)
