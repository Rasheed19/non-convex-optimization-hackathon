import click

from pipelines import eda_pipeline, training_pipeline
from utils.constants import ROOT_DIR
from utils.helper import load_yaml_file


@click.command(
    help="""
    Entry point for running all pipelines.
    """
)
@click.option(
    "--exp-name",
    default="default-exp",
    help="""Experiment name to be given to
    comet-ml experiment tracker handler.
    Default to 'default-exp'.
        """,
)
@click.option(
    "--only-eda",
    is_flag=True,
    default=False,
    help="""If given, only eda pipeline will be 
    run.
        """,
)
@click.option(
    "--device",
    default="cpu",
    help="""Device to run pipeline on. Valid options
    are 'cpu' and 'cuda'.
    Default to 'cpu'.
        """,
)
@click.option(
    "--optimizer-name",
    default="adam",
    help="""Optimzer to use for training model. Valid options
    are 'adam', 'sgd'...
    Default to 'adam'.
        """,
)
@click.option(
    "--batch-size",
    default=32,
    type=click.IntRange(min=1, max=10000),
    help="""Specify the batch size for model training.
    Default to 32.
        """,
)
@click.option(
    "--epochs",
    default=2,
    type=click.IntRange(min=1, max=500),
    help="""Specify how many epochs to train model.
    Default to 1.
        """,
)
def main(
    exp_name: str = "default-exp",
    only_eda: bool = False,
    device: str = "cpu",
    optimizer_name: str = "adam",
    batch_size: int = 32,
    epochs: int = 1.0,
) -> None:

    MODEL_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/configs/model_config.yaml")

    if only_eda:
        eda_pipeline()

        return None

    training_pipeline(
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        optimizer_params=MODEL_CONFIG["optimizers"][optimizer_name],
        epochs=epochs,
        device=device,
        exp_name=exp_name,
    )

    return None


if __name__ == "__main__":
    main()


# TODO:
# address all the # FIXME:
# remove run and Optim branches; only legacy and main branch should be left
# update README and clean up REPO
# Poster? I have template!!!
