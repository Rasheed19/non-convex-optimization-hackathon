import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import logging
import yaml


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Times"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
    }

    return rc_params


def get_image_meta_data(path_to_split: str) -> pd.DataFrame:
    image_path_list = open(path_to_split, "r").read().splitlines()

    # Getting the full path for the images
    full_path = ["./data/food-101/images/" + img + ".jpg" for img in image_path_list]

    # Splitting the image index from the label
    image_index_label = []
    for img in image_path_list:
        split = img.split("/")
        image_index_label.append(split)

    image_index_label = np.array(image_index_label)
    # Converting the array to a data frame
    image_index_label = pd.DataFrame(
        image_index_label[:, 0], image_index_label[:, 1], columns=["label"]
    )
    # Adding the full path to the data frame
    image_index_label["path"] = full_path

    # Randomly shuffling the order to the data in the dataframe
    image_index_label = shuffle(image_index_label)

    return image_index_label


class CustomFormatter(logging.Formatter):

    green = "\x1b[0;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def config_logger() -> None:

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )


def load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data
