import os

ROOT_DIR: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR: str = f"{ROOT_DIR}/models"
PLOT_DIR: str = f"{ROOT_DIR}/plots"
DATA_DIR: str = f"{ROOT_DIR}/data/food-101"
NUM_CLASSES: int = 101
IMAGE_DIMENSION: tuple[int] = (3, 224, 224)
