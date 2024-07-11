import torch
from torch import nn
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models
import os
from dotenv import load_dotenv
from comet_ml import Experiment
from typing import Any

from utils.constants import NUM_CLASSES, IMAGE_DIMENSION, MODEL_DIR
from utils.optimizer import (
    get_optimizer,
    StepLR,
    SAM,
    GSAM,
    CosineScheduler,
    ProportionScheduler,
)
from utils.extras import (
    training_base,
    sam_single_epoch_training,
    gsam_single_epoch_training,
)


load_dotenv()


def create_experiment(exp_name: str = "exp_02") -> Experiment:
    API_KEY = os.getenv("API_KEY")
    WORKSPACE = os.getenv("WORKSPACE")

    exp = Experiment(project_name="hackathon", api_key=API_KEY, workspace=WORKSPACE)
    exp.set_name(exp_name)

    return exp


def create_model() -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    classifier = nn.Sequential(
        nn.Linear(in_features=model.last_channel, out_features=1024),
        nn.LeakyReLU(),
        nn.Linear(in_features=1024, out_features=NUM_CLASSES),
    )
    model.classifier = classifier

    print(summary(model, IMAGE_DIMENSION))

    return model


def single_epoch_training(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    comet_experiment_tracker_handler: Experiment,
    scheduler: Any | None = None,
) -> tuple[float]:

    model.train()

    if isinstance(optimizer, SAM) and scheduler is not None:
        train_loss, train_acc = sam_single_epoch_training(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            comet_experiment_tracker_handler=comet_experiment_tracker_handler,
        )

    elif isinstance(optimizer, GSAM) and scheduler is not None:
        train_loss, train_acc = gsam_single_epoch_training(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            comet_experiment_tracker_handler=comet_experiment_tracker_handler,
        )

    else:
        train_loss, train_acc = training_base(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            comet_experiment_tracker_handler=comet_experiment_tracker_handler,
        )

    return train_loss, train_acc


def single_epoch_testing(
    model: nn.Module,
    test_data: DataLoader,
    loss_function: nn.Module,
    device: str,
    epoch: int,
    comet_experiment_tracker_handler: Experiment,
) -> tuple[float]:

    model.eval()

    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for i, (X, y) in enumerate(test_data):
            images, labels = X.to(device), y.to(device)

            test_pred_logits = model(images)

            loss = loss_function(test_pred_logits, labels)
            test_loss += loss.item()

            test_pred_labels = torch.argmax(
                torch.softmax(test_pred_logits, dim=1), dim=1
            )

            test_acc += (test_pred_labels == labels).sum().item() / len(
                test_pred_labels
            )

            comet_experiment_tracker_handler.log_metric(
                "test_loss_in_epoch", test_loss, step=i + 1, epoch=epoch + 1
            )

    test_loss = test_loss / len(test_data)
    test_acc = test_acc / len(test_data)

    return test_loss, test_acc


def model_trainer(
    train_data: DataLoader,
    test_data: DataLoader,
    batch_size: int,
    optimizer_name: str,
    optimizer_params: dict,
    epochs: int,
    device: str,
    exp_name: str,
    loss_function: nn.Module = nn.CrossEntropyLoss(),
) -> tuple[nn.Module, dict]:

    model = create_model().to(device)
    comet_experiment_tracker_handler = create_experiment(exp_name=exp_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")
        model = nn.DataParallel(model)

    if optimizer_name in ["adam", "sgd", "sgld", "sam"]:
        optimizer_params["params"] = model.parameters()
        optimizer = get_optimizer(
            optimizer_name=optimizer_name, optimizer_params=optimizer_params
        )

        if optimizer_name == "sam":
            scheduler = StepLR(
                optimizer=optimizer,
                learning_rate=optimizer_params["lr"],
                total_epochs=epochs,
            )  # FIXME: I am not sure what this schedular does of SAM implementation
        else:
            scheduler = None

    elif optimizer_name == "gsam":
        # base_optimizer = SGD(
        #     model.parameters(),
        #     lr=optimizer_params["lr"],
        #     momentum=optimizer_params["momentum"],
        #     weight_decay=optimizer_params["weight_decay"],
        # )

        base_optimizer = Adam(
            model.parameters(),
            lr=optimizer_params["lr"],
        )

        scheduler_ = CosineScheduler(
            T_max=epochs * len(train_data),
            max_value=optimizer_params["lr"],
            min_value=0.0,
            optimizer=base_optimizer,
        )
        rho_scheduler = ProportionScheduler(
            pytorch_lr_scheduler=scheduler_,
            max_lr=optimizer_params["lr"],
            min_lr=0.0,
            max_value=optimizer_params["rho_max"],
            min_value=optimizer_params["rho_min"],
        )
        scheduler = (scheduler_, rho_scheduler)

        optimizer = GSAM(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            gsam_alpha=optimizer_params["alpha"],
            rho_scheduler=scheduler[1],
            adaptive=optimizer_params["adaptive"],
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    with comet_experiment_tracker_handler.train():
        exp_params = {
            "neural_network": {
                "type": "mobilenet_v2(weights=None)",
                "batch_size": batch_size,
                "num_gpus": 3,
                "num_trainable_params": "~3,600,000",
            },
            "optimizer": {
                "type": optimizer_name,
                "optimizer_params": optimizer_params,
            },
            "loss_function": {
                "type": "nn.CrossEntropyLoss()",
            },
        }
        comet_experiment_tracker_handler.log_parameters(parameters=exp_params)

    for epoch in range(epochs):

        train_loss, train_acc = single_epoch_training(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            comet_experiment_tracker_handler=comet_experiment_tracker_handler,
            scheduler=scheduler,
        )
        test_loss, test_acc = single_epoch_testing(
            model=model,
            test_data=test_data,
            loss_function=loss_function,
            device=device,
            epoch=epoch,
            comet_experiment_tracker_handler=comet_experiment_tracker_handler,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"train_accuracy: {train_acc:.4f} | "
            f"test_accuracy: {test_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)

        comet_experiment_tracker_handler.log_metric(
            "train_loss_end_epoch", train_loss, epoch=epoch + 1
        )
        comet_experiment_tracker_handler.log_metric(
            "test_loss_end_epoch", test_loss, epoch=epoch + 1
        )
        comet_experiment_tracker_handler.log_metric(
            "train_accuracy_end_epoch", train_acc, epoch=epoch + 1
        )
        comet_experiment_tracker_handler.log_metric(
            "test_accuracy_end_epoch", test_acc, epoch=epoch + 1
        )

    torch.save(
        obj=model.state_dict(),
        f=f"{MODEL_DIR}/model_batch_size={batch_size}_optimizer={optimizer_name}_epochs={epochs}.pt",
    )
    return model, history
