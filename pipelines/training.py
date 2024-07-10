import logging

from steps import data_loader, model_trainer
from utils.helper import config_logger, sample_dataloader
from utils.plotter import plot_training_pipeline_history


def training_pipeline(
    optimizer_name: str, optimizer_params: dict, epochs: int, device: str
) -> None:

    logger = logging.getLogger(__name__)
    config_logger()

    train_data, test_data = data_loader()

    # # # FIXME: just to test the pipeline; remove later
    train_data = sample_dataloader(train_data, 100)
    test_data = sample_dataloader(test_data, 50)

    model, history = model_trainer(
        train_data=train_data,
        test_data=test_data,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
        epochs=epochs,
        device=device,
    )
    logger.info("Plotting training history...")
    plot_training_pipeline_history(
        history=history, epochs=epochs, optimizer_name=optimizer_name
    )

    logger.info(
        "Training pipeline has finished successfully. "
        "Check the 'plots' folder for the generated plots."
    )

    return None
