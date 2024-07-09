import os

ROOT_DIR: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_DIR: str = f"{ROOT_DIR}/plots"
DATA_DIR: str = f"{ROOT_DIR}/data/food-101"
NUM_CLASSES: int = 101
