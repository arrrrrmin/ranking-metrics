from typing import Any

from faiss import IndexFlat
import numpy as np
import torch
from torch import nn, Tensor

from ranking_metrics.metrics import (
    sizes_per_class,
    conf_vs_recall_at_1,
    erc_vs_recall_at,
    erc_vs_mapr, spearmanr_corr,
)


class UncertaintyMetrics(nn.Module):
    def __init__(self, dim, r=None) -> None:
        super(UncertaintyMetrics, self).__init__()
        self.dim = dim
        self.r = r
        self.knn_index = IndexFlat(dim)

    @staticmethod
    def get_binary_hits(c, inds) -> Tensor:
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

        return torch.from_numpy(binary_hits).float()

    def forward(self, d, c, confs, gt_confs, k, ignore_self=True) -> dict[str, Any]:
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
        results_dict["conf_vs_recall@1"] = conf_vs_recall_at_1(binary_matches, confs)
        results_dict["rcorr"] = spearmanr_corr(confs, gt_confs)
        results_dict[f"erc_vs_recall@{k}"] = erc_vs_recall_at(binary_matches, confs, k)
        results_dict[f"erc_vs_map@r"] = erc_vs_mapr(binary_matches, confs, r - ignore_self)

        return results_dict
