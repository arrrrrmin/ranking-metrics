from typing import Any

from faiss import IndexFlat
import numpy as np
import torch
from torch import nn

from ranking_metrics.metrics import recall_at, mean_average_precision_at_r, r_precision, sizes_per_class


class ClassBasedEmbeddingMetrics(nn.Module):
    def __init__(self, dim, ks=(1, 5, 10), r=None) -> None:
        """Class to compute multi-class based embedding metrics.
        This is useful for contrastivly learned latent spaces.
        To compute recall@k metrics pass a list or tuple of ints for different
        k values. The r value is used to indicate R for R-Precision (MAP@R).
        R determines the number of ranked trials per sample in
        Musgrave et al. 2020. "A Metric Learning Reality Check"

        :param ks:  A list or tuple of ints for each recall@k metric to calculate. E.g.: (1, 5, 10)
        :param r:   If None r is found by the class with most samples, else r is set as maximum
                    rank to compute MAP@R.
        """
        super(ClassBasedEmbeddingMetrics, self).__init__()
        self.dim = dim
        self.ks = ks
        self.r = r
        self.knn_index = IndexFlat(dim)

    @staticmethod
    def get_binary_hits(c, inds) -> np.ndarray:
        """

        :param c: Classes/labels to map embedding vectors to source classes. Dims: (n,)
        :param inds: Indices of nearest neighbors per sample.
        :return: Binary hits per sample and nearest neighbors. Dims: (n, r)
        """
        binary_hits = []
        for i, idx in enumerate(inds):
            # Check if the class of nearest neighbors are correct (hit)
            hits = np.array(c[idx] == c[i]).astype("int8")
            binary_hits.append(hits)
        binary_hits = np.stack(binary_hits)

        return binary_hits

    def forward(self, d, c, ignore_self=True) -> dict[str, Any]:
        """Calculates recall@{k1, k2, ... kn}, r-precision and MAP@R for the embedding space d.

        :param d: Embedding space based on some network or algorithm. Dims: (n, z_dim)
        :param c: Classes/labels to map embedding vectors to source classes. Dims: (n,)
        :param ignore_self: Wether to ignore the queried vector itself or not. Default: True.
        :return: A dictionary for each "recall@{k1, k2, ... kn}" and "MAP@R". Values are floats.
        """

        self.knn_index.add(d)  # noqa
        c_sizes = sizes_per_class(c)
        r = int(min(c_sizes)) - 1 if self.r is None else self.r
        r = r + 1 if ignore_self else r
        _, inds = self.knn_index.search(d, r)  # noqa
        inds = torch.from_numpy(inds)
        self.knn_index.reset()  # We are done with searching, reset the index

        # Remove self, assuming no nearest neighbor can't be more near than query itself.
        if ignore_self:
            _ = _[:, 1:]
            inds = inds[:, 1:]

        results_dict = {}
        binary_matches = self.get_binary_hits(c, inds)

        # Compute precision@K entries
        for k in self.ks:
            results_dict[f"recall@{k}"] = recall_at(binary_matches, k).mean().item()

        # Compute precision@K entries
        rprec = r_precision(binary_matches, c)
        mapr = mean_average_precision_at_r(binary_matches, r - ignore_self)
        results_dict["r-precision"] = rprec.mean()
        results_dict["map@r"] = mapr.mean()

        return results_dict
