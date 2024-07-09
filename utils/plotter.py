import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.constants import PLOT_DIR


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
