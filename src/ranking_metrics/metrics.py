import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve
from torch import Tensor


def sizes_per_class(c) -> Tensor:
    # Each class c corresponds to the number of samples with this class c_n.

    return torch.tensor([len(c[c == cid]) for cid in range(c.max() + 1)]).long()


def order_by_confidence(t, confs, descending=True) -> Tensor:
    # Sort confidences from high to low by default.
    order = torch.argsort(confs, descending=descending)

    return t[order]


def mean_average_precision_at_r(binary_matches, r) -> Tensor:
    # Calculate MAP@R using binary matches of query classes and r, which is the
    # least large class size. The r parameter can be found by using 'min(get_class_sizes(c))'.
    # This method is a slightly changed version of the one used by:
    # https://github.com/tinkoff-ai/probabilistic-embeddings/blob/main/src/probabilistic_embeddings/metrics/nearest.py
    arange = torch.arange(1, r + 1).tile(binary_matches.shape[0], 1)
    precisions = binary_matches * binary_matches.cumsum(1) / arange
    maprs = precisions.sum(-1) / r

    return maprs


def r_precision(binary_matches, c) -> Tensor:
    # Calculate r-precision where for every step in r the score is calculated by r/R,
    # where r is the found matches up to R and R is the total number of that class appearing.
    s_per_class = sizes_per_class(c)
    class_sizes = torch.tensor([s_per_class[c_] for c_ in c])
    precisions = binary_matches.sum(-1) / class_sizes

    return precisions


def recall_at(binary_matches, k, aggregate=False) -> Tensor:
    # Calculate the recall@k metric.
    recalls = binary_matches[:, :k].max(1)[0]
    if aggregate:
        recalls = recalls.mean()

    return recalls


def conf_vs_recall_at_1(binary_matches, confs) -> float:
    # Compute precision form the first neighbors per sample
    matches_at_1 = recall_at(binary_matches, 1)
    prec = matches_at_1.float().mean().item()
    fprs, tprs, ths = roc_curve(
        matches_at_1.long().cpu().numpy(),
        confs.cpu().numpy(),
        drop_intermediate=False,
    )
    # Not sure if averaging here is the right way to do it (todo).
    confidence_recall_accuracy = (prec * tprs + (1 - prec) * (1 - fprs)).mean()

    return confidence_recall_accuracy


def error_vs_reject_curve(binary_hits, scores, confs) -> float:
    # Compute the curve between error and rejection by confidences.
    errors = 1 - scores.float()
    errors = order_by_confidence(errors, confs)
    cum_errors = errors.cumsum(0) / torch.arange(1, binary_hits.shape[0] + 1)

    return cum_errors.mean().item()


def erc_vs_recall_at(binary_matches, confs, k) -> float:
    # Compute error reject curve with recall@k as comparison metric.
    recalls, _ = binary_matches[:, :k].max(1)

    return error_vs_reject_curve(binary_matches, recalls, confs)


def erc_vs_mapr(binary_matches, confs, r) -> float:
    # Compute error reject curve using the mean average precision at r metric.
    maprs = mean_average_precision_at_r(binary_matches, r)

    return error_vs_reject_curve(binary_matches, maprs, confs)


def spearmanr_corr(confs, gt_confs) -> float:
    # Just to have it in this file (simply use scikit-learns function.
    rcorr, _ = spearmanr(confs.numpy(), gt_confs.numpy())  # Omit p-value

    return rcorr
