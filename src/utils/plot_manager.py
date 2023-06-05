import matplotlib.pyplot as plt

FIG_SIZE = (12, 8)


def plot_history(history: dict[str, list], metrics: list[str]) -> None:
    plt.figure(figsize=FIG_SIZE)

    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        plt.plot(history[metric])
        plt.plot(history[f"val_{metric}"])

        plt.title(f"{metric.upper()} HISTORY")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="best")
