from faiss import IndexFlat
import numpy as np
import torch
from torch import nn


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

    def forward(self, d, c, ignore_self=True):
        """Calculates recall@{k1, k2, ... kn} and MAP@R for the embedding space d.

        :param d:   Embedding space based on some network or algorithm. Dims: (n, z_dim)
        :param c:   Classes/labels to map embedding vectors to source classes. Dims: (n,)
        :param ignore_self: Wether to ignore the queried vector itself or not. Default: True.
        :return: A dictionary for each "recall@{k1, k2, ... kn}" and "MAP@R". Values are floats.
        """

        self.knn_index.add(d)  # noqa
        c_sizes = self._get_class_sizes(c)
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
        binary_hits = []
        for i, idx in enumerate(inds):
            # Check if the class of nearest neighbors are correct (hit)
            hits = np.array(c[idx] == c[i]).astype("int8")
            binary_hits.append(hits)
        binary_hits = np.stack(binary_hits)

        # Compute precision@K entries
        for k in self.ks:
            results_dict[f"recall@{k}"] = (binary_hits[:, :k].sum(-1) / k).mean()

        mapr_classwise, r_precisions = [], []
        for c_idx, _ in enumerate(c_sizes):
            # Samples that correspond to this class
            class_mask = c == c_idx

            # Add the R-Precision for this class r/R (r beeing number
            # of matching nearest neighbors and R the number of samples
            # corresponding to observed class)
            r_precisions.append(binary_hits[class_mask, : sum(class_mask)].mean())

        mapr_classwise = np.array(
            [
                np.array(
                    [
                        hit[: r_idx + 1].mean() if hit[r_idx] == 1 else 0.0
                        for r_idx in range(0, r - 1)
                    ]
                ).mean()
                for i, hit in enumerate(binary_hits)
            ]
        )
        r_precisions = np.stack(r_precisions, axis=0)
        # Now average over all classes
        results_dict["map@r"] = mapr_classwise.mean()
        results_dict["r-precision"] = r_precisions.mean()

        return results_dict

    @staticmethod
    def _get_class_sizes(c) -> np.ndarray:
        """Private function to find the sizes of each class in c.
        Each class c corresponds to the number of samples with this class c_n.
        Also very useful to find R per class when self.r is None.

        :return: Numpy array containing the number of samples per class. Dim: (c, n_c)
        """
        return np.array([len(c[c == cid]) for cid in range(c.max() + 1)])
