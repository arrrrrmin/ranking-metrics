import pytest
import torch
from torch import Tensor, Size
from torch.distributions import Uniform


torch.manual_seed(42)


def quarter(n) -> Size:
    return Size((n // 4, 1))


def half(n) -> Size:
    return Size((n // 2, 1))


@pytest.fixture
def n() -> int:
    return 2000


@pytest.fixture
def musgrave_example_1(n) -> tuple[Tensor, Tensor]:
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-1.0, -(2 / 3))
    x_dist_2 = Uniform(-(1 / 3), 0.0)
    x_dist_3 = Uniform(0.0, 1 / 3)
    x_dist_4 = Uniform(2 / 3, 1.0)
    d1 = torch.cat([x_dist_1.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d3 = torch.cat([x_dist_3.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d4 = torch.cat([x_dist_4.sample(quarter(n)), y_dist.sample(quarter(n))], dim=-1)
    d = torch.cat([d1, d2, d3, d4], dim=0)
    c = torch.zeros((d.shape[0],)).long()
    c[d[:, 0] > 0.0] = 1

    return d, c


@pytest.fixture
def musgrave_example_2(n) -> tuple[Tensor, Tensor]:
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-1.0, 0.0)
    x_dist_2 = Uniform(0.0, 1.0)
    d1 = torch.cat([x_dist_1.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d = torch.cat([d1, d2], dim=0)
    c = torch.zeros((d.shape[0],)).long()
    c[d[:, 0] > 0.0] = 1

    return d, c


@pytest.fixture
def musgrave_example_3(n) -> tuple[Tensor, Tensor]:
    y_dist = Uniform(0.0, 1.0)
    x_dist_1 = Uniform(-1.0, -(1 / 3))
    x_dist_2 = Uniform(1 / 3, 1.0)
    d1 = torch.cat([x_dist_1.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d2 = torch.cat([x_dist_2.sample(half(n)), y_dist.sample(half(n))], dim=-1)
    d = torch.cat([d1, d2], dim=0)
    c = torch.zeros((d.shape[0],)).long()
    c[d[:, 0] > 0.0] = 1

    return d, c
