from matplotlib import pyplot as plt

from src.ranking_metrics.embed_metrics import ClassBasedEmbeddingMetrics


def _example_helper(d, c, example_id, ks=(1, 10)) -> None:
    print(f"Number of samples used in exmaple {example_id}: {d.shape[0]}")
    m = ClassBasedEmbeddingMetrics(d.shape[-1], ks)
    scores = m.forward(d, c, ignore_self=True)

    cc = ["#219ebcff", "#ffb703ff"]
    print_scores = ", ".join((f"{k}: {(float(v) * 100):.1f}%" for k, v in scores.items()))

    _ = plt.figure(figsize=(10, 7))
    plot_title = f"Example {example_id} of Musgrave et al. 2020. \n {print_scores}"
    plt.title(plot_title)
    plt.scatter(d[:, 0], d[:, 1], c=[cc[int(c_id)] for c_id in c])
    plt.savefig(f"figures/reality_check_example{example_id}.png")
    plt.close()


def test_class_reality_check_example_1(musgrave_example_1):
    d, c = musgrave_example_1
    _example_helper(d, c, 1)


def test_class_reality_check_example_2(musgrave_example_2):
    d, c = musgrave_example_2
    _example_helper(d, c, 2)


def test_class_reality_check_example_3(musgrave_example_3):
    d, c = musgrave_example_3
    _example_helper(d, c, 3)
