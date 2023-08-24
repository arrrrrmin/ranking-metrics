import faiss
import numpy as np
from icecream import ic

from src.ranking_metrics.embed_metrics import ClassBasedEmbeddingMetrics


def _test_class_based_embedding_metrics():
    np.random.seed(42)
    d_size, n_classes, dim = 10000, 5, 64
    d = np.random.random((d_size, dim)).astype("float32")
    c = np.random.randint(low=0, high=n_classes, size=(d_size,))
    m = ClassBasedEmbeddingMetrics(dim, (1, 5, 10, 25, 100))
    assert all([c > 0 for c in m.c_sizes])
    _ = m.forward(d, c, ignore_self=True)


# Optionally view inputs and outputs of faiss
def _test_faiss():
    np.random.seed(42)
    d_size, q_size, dim = 100000, 10000, 64
    d = np.random.random((d_size, dim)).astype("float32")
    d[:, 0] += np.arange(d_size) / 1000.0  # (d_size, dim)
    q = np.random.random((q_size, dim)).astype("float32")
    q[:, 0] += np.arange(q_size) / 1000.0  # (q_size, dim)

    # ic(d.shape)
    # ic(q.shape)

    index = faiss.IndexFlatL2(dim)
    # ic(index.is_trained)
    index.add(d)  # noqa
    # ic(index.ntotal)

    k = 4
    D, I = index.search(d[:5], k)  # sanity check # noqa
    # ic(D)
    # ic(I)

    assert D.shape == (5, k)
    assert I.shape == (5, k)
    D, I = index.search(q, k)  # actual search # noqa

    assert D.shape == (q.shape[0], k)
    assert I.shape == (q.shape[0], k)

    # ic(I[:5])  # neighbors of the 5 first queries
    # ic(I[-5:])  # neighbors of the 5 last queries


# Optionally illustrate inputs and outputs of faiss
def _test_multiclass_scenario():
    d_size, dim = 100000, 64
    n_classes = 10
    d = np.random.random((d_size, dim)).astype("float32")
    c = np.random.randint(low=0, high=n_classes, size=(d_size,))
    d[:, 0] += np.arange(d_size) / 1000.0  # (d_size, dim)
    index = faiss.IndexFlatL2(dim)
    index.add(d)  # noqa
    k = 5

    D, I = index.search(d, k)  # actual search # noqa
    C = np.array([c[inds] for inds in I])

    assert D.shape == (d_size, k)
    assert I.shape == (d_size, k)
    assert C.shape == (d_size, k)

    ic(D.shape)
    ic(I.shape)
    ic(C.shape)
    ic(C)
