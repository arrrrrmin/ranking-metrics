import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, Size
from torch.distributions import Normal, Uniform, Distribution, MultivariateNormal, Pareto

from ranking_metrics.metrics import spearmanr_corr
from src.ranking_metrics.uncertainty import UncertaintyMetrics


cc = ["#219ebcff", "#ffb703ff"]


def simulate_confidences(d, good_quality) -> tuple[Tensor, Distribution]:
    n_dist = Normal(0.0, 0.5)
    confs = torch.exp(n_dist.log_prob(d[:, 0]))
    if good_quality:
        confs = 1.0 - confs
    return confs, n_dist


def _test_confidence_example_2_basic(musgrave_example_2):
    d, c = musgrave_example_2
    example_id = 2
    confs, _ = simulate_confidences(d, True)
    dist = Uniform(0, d.shape[0])  # Distribution to sample sample ids for error
    corrupt_inds = dist.rsample(Size((100,))).long()
    # Invert some ground truth samples
    c[corrupt_inds] = torch.tensor([1 if c[ind] == 0 else 0 for ind in corrupt_inds])
    # Simulate some confidences and manipulate the erroneous samples
    corrupted_confs = 1.0
    confs[corrupt_inds] = corrupted_confs

    _ = plt.figure(figsize=(10, 7))
    plot_title = (
        f"Example {example_id} of Musgrave et al. 2020 with simulated confidences and errors."
    )
    plt.title(plot_title)
    plt.scatter(d[:, 0], d[:, 1], c=[cc[int(c_id)] for c_id in c], alpha=confs)
    plt.savefig(f"figures/confidence_example{example_id}.png")
    plt.close()


def test_uncertainty_example_2(musgrave_example_2):
    # Extended version of the Musgrave et al. example nr. 2 (most easy to simulate)
    d, c = musgrave_example_2
    k = 1
    metrics = UncertaintyMetrics(d.shape[-1])
    dist = Uniform(0, d.shape[0])  # Distribution to sample sample ids for error
    corrupt_inds = dist.rsample(Size((100,))).long()
    # Invert some ground truth samples
    c[corrupt_inds] = torch.tensor([1 if c[ind] == 0 else 0 for ind in corrupt_inds])

    # Simulate some confidences and manipulate the erroneous samples
    # multiple times with a linear increasing confidence on corrupted samples.
    n_confs = 20
    scores = np.zeros((3, n_confs, 2))
    gt_confs, _ = simulate_confidences(d, True)
    confs = gt_confs.clone()
    for i in range(1, n_confs + 1):
        corrupted_confs = float(i) / n_confs
        confs[corrupt_inds] = corrupted_confs

        result_dict = metrics(d, c, confs, gt_confs, k)
        scores[0, i - 1, 0] = corrupted_confs
        scores[0, i - 1, 1] = result_dict["conf_vs_recall@1"]
        scores[1, i - 1, 0] = corrupted_confs
        scores[1, i - 1, 1] = result_dict[f"erc_vs_recall@{k}"]
        scores[2, i - 1, 0] = corrupted_confs
        scores[2, i - 1, 1] = result_dict[f"erc_vs_map@r"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # todo title
    cc = ["#219ebcff", "#ffb703ff"]
    labels = [
        f"Confidence vs. Recall@{1}",
        f"Error vs. reject curve via Recall@{k}",
        "Error vs. reject curve via MAP@R",
    ]
    ax1.plot(scores[0, :, 0], scores[0, :, 1], label=labels[0], c=cc[0])
    ax1.set_xlabel("Confidence on error samples")
    ax1.set_ylabel("Score", color=cc[0])
    ax1.tick_params(axis="y", labelcolor=cc[0])

    ax2 = ax1.twinx()
    ax2.plot(scores[1, :, 0], scores[1, :, 1], label=labels[1], c=cc[1])
    ax2.set_ylabel("Error", color=cc[1])
    ax2.tick_params(axis="y", labelcolor=cc[1])

    ax2.plot(scores[2, :, 0], scores[2, :, 1], label=labels[2], c=cc[1], linestyle="--")

    fig.legend()
    fig.tight_layout()
    plt.savefig(f"figures/confidence_recall_metrics.png")
    plt.close()


def test_uncertainty_example_2_spearman(musgrave_example_2):
    d, c = musgrave_example_2
    gt_confs, conf_dist = simulate_confidences(d, True)
    sig_scale = 12500  # Some arbitrary number to simulate models scale on sigma
    confs = gt_confs.clone() * sig_scale

    rcorr = spearmanr_corr(confs, gt_confs)
    assert rcorr == 1.0

    # Manipulate some confidence values
    dist = Uniform(0, d.shape[0])  # Distribution to sample sample ids for error
    corrupt_inds = dist.rsample(Size((100,))).long()

    n_confs = 20
    scores, corrupt_prob_confs = [], []
    for i in range(1, n_confs + 1):
        corrupt_prob_conf = float(i) / n_confs
        corrupt_prob_confs.append(corrupt_prob_conf * sig_scale)
        confs[corrupt_inds] = corrupt_prob_conf * sig_scale
        rcorr = spearmanr_corr(confs, gt_confs)
        scores.append(rcorr)

    fig, _ = plt.subplots(figsize=(10, 6))

    labels = ["Rank correlation", "Simulated conf probabilities"]
    plt_title = (
        "Spearman corr. between confidences and scaled model $\hat{\sigma}$ \n $\hat{\sigma}$ scaled by "
        + str(sig_scale)
    )
    plt.title(plt_title)
    plt.plot(corrupt_prob_confs, scores, label=labels[0], c=cc[0])
    plt.xlabel("Confidence on error samples")
    plt.ylabel("Score")

    original_confs_avrg = round(gt_confs[corrupt_inds].mean().item(), 4) * sig_scale
    plt.axvline(
        x=original_confs_avrg,
        label=f"Avrg. of original confidences ~ {original_confs_avrg}",
        c=cc[1],
        linestyle="--",
    )

    plt.legend()
    plt.savefig("figures/rank_corr_over_simulated_sigmas.png")
