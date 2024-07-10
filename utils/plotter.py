import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.constants import PLOT_DIR
from utils.helper import get_rcparams

plt.rcParams.update(get_rcparams())


def set_size(
    width: float | str = 360.0, fraction: float = 1.0, subplots: tuple = (1, 1)
) -> tuple[float, float]:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_food_samples(image_meta_data: pd.DataFrame) -> None:

    plt.figure(figsize=(20, 5))

    num_rows = 3
    num_cols = 8

    for idx in range(num_rows * num_cols):
        random_idx = np.random.randint(0, image_meta_data.shape[0])
        img = plt.imread(image_meta_data.path.iloc[random_idx])

        label = image_meta_data.label.iloc[random_idx]

        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(fname=f"{PLOT_DIR}/food_samples.pdf", bbox_inches="tight")

    return None


def plot_training_pipeline_history(
    history: dict, epochs: int, optimizer_name: str
) -> None:

    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ["a", "b"]
    epoch_array = np.arange(epochs) + 1

    for i, m in enumerate(["loss", "accuracy"]):
        ax = fig.add_subplot(1, 2, i + 1)
        # ax.text(
        #     x=-0.1,
        #     y=1.2,
        #     s=r"\bf \large {}".format(fig_labels[i]),
        #     transform=ax.transAxes,
        #     fontweight="bold",
        #     va="top",
        # )

        for d, c in zip(["train", "test"], ["black", "fuchsia"]):
            ax.plot(
                epoch_array,
                history[f"{d}_{m}"],
                color=c,
                linewidth=2,
                label=f"{d}".capitalize(),
            )
        ax.set_xlabel("Epochs")
        ax.set_ylabel(m.capitalize())

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.savefig(
        fname=f"{PLOT_DIR}/training_history_plot_optimizer={optimizer_name}_epochs={epochs}.pdf",
        bbox_inches="tight",
    )

    return None
