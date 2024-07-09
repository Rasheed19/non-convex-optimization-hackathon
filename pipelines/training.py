import logging

from steps import data_loader, model_trainer
from utils.helper import config_logger


def training_pipeline(
    optimizer_name: str, optimizer_params: dict, epochs: int, device: str
) -> None:

    logger = logging.getLogger(__name__)
    config_logger()

    training_data, test_data = data_loader()

    _ = model_trainer(
        training_data=training_data,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
        epochs=epochs,
        device=device,
    )

    logger.info("Training pipeline has finished successfully.")

    return None
