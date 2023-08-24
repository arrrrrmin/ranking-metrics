import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch import Size
from torch.distributions import Uniform

from src.ranking_metrics.embed_metrics import ClassBasedEmbeddingMetrics


def quarter(n) -> Size:
    return Size((n // 4, 1))


def half(n) -> Size:
    return Size((n // 2, 1))


@pytest.fixture
def n() -> int:
    return 2000


def _example_helper(d, c, example_id, ks=(1, 10)):
    print(f"Number of samples used in exmaple {example_id}: {d.shape[0]}")
    m = ClassBasedEmbeddingMetrics(d.shape[-1], ks)
    scores = m.forward(d, c, ignore_self=False)

    cc = ["#8ecae6ff", "#ffb703ff"]
    print_scores = ", ".join((f"{k}: {(float(v) * 100):.1f}%" for k, v in scores.items()))

    _ = plt.figure(figsize=(10, 7))
    plot_title = f"Example {example_id} of Musgrave et al. 2020. \n {print_scores}"
    plt.title(plot_title)
    plt.scatter(d[:, 0], d[:, 1], c=[cc[int(c_id)] for c_id in c])
    plt.savefig(f"figures/reality_check_example{example_id}.png")
    plt.close()


def test_class_reality_check_example_1(n):
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-3.0, -2.0)
    x_dist_2 = Uniform(-1.0, 0.0)
    x_dist_3 = Uniform(0.0, 1.0)
    x_dist_4 = Uniform(2.0, 3.0)
    d1 = torch.cat([x_dist_1.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d3 = torch.cat([x_dist_3.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d4 = torch.cat([x_dist_4.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d = torch.cat([d1, d2, d3, d4], dim=0)
    c = np.zeros((d.shape[0],), dtype="uint8")
    c[d[:, 0] > 0.0] = 1
    _example_helper(d, c, 1)


def test_class_reality_check_example_2(n):
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-1.0, 0.0)
    x_dist_2 = Uniform(0.0, 1.0)
    d1 = torch.cat([x_dist_1.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d = torch.cat([d1, d2], dim=0)
    c = np.zeros((d.shape[0],), dtype="uint8")
    c[d[:, 0] > 0.0] = 1
    _example_helper(d, c, 2)


def test_class_reality_check_example_3(n):
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-3.0, -1.0)
    x_dist_2 = Uniform(1.0, 3.0)
    d1 = torch.cat([x_dist_1.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d = torch.cat([d1, d2], dim=0)
    c = np.zeros((d.shape[0],), dtype="uint8")
    c[d[:, 0] > 0.0] = 1
    _example_helper(d, c, 3)
