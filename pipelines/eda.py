from utils.helper import get_image_meta_data, config_logger
from utils.plotter import plot_food_samples
from utils.constants import DATA_DIR
import logging


def eda_pipeline() -> None:

    config_logger()
    logger = logging.getLogger(__name__)

    train_image_meta_data = get_image_meta_data(
        path_to_split=f"{DATA_DIR}/meta/train.txt"
    )

    logger.info("Getting training data summary...")
    print(train_image_meta_data["label"].value_counts(sort=True))
    print(f"Total number of train samples: {train_image_meta_data.shape[0]}")

    logger.info("Plotting sample food images...")

    plot_food_samples(image_meta_data=train_image_meta_data)

    logger.info(
        "EDA pipeline finished sucessfully. "
        "Check the plots folder for the generated images."
    )

    return None
