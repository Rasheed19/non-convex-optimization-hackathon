import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any
from comet_ml import Experiment

from utils.optimizer import (
    enable_running_stats,
    disable_running_stats,
    StepLR,
)


def training_base(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    comet_experiment_tracker_handler: Experiment,
) -> tuple[float]:

    train_loss, train_acc = 0.0, 0.0

    for i, (X, y) in enumerate(train_data):
        images, labels = X.to(device), y.to(device)

        y_pred = model(images)

        loss = loss_function(y_pred, labels)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(y_pred)

        comet_experiment_tracker_handler.log_metric(
            "train_loss_in_epoch", train_loss, step=i + 1, epoch=epoch + 1
        )

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss, train_acc


def sam_single_epoch_training(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    scheduler: StepLR,
    comet_experiment_tracker_handler: Experiment,
) -> tuple[float]:

    train_loss, train_acc = 0.0, 0.0

    for i, (X, y) in enumerate(train_data):
        images, labels = X.to(device), y.to(device)
        # first forward-backward step
        enable_running_stats(model=model)

        y_pred = model(images)

        loss = loss_function(y_pred, labels)
        train_loss += loss.item()

        loss.backward()

        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model=model)
        loss_function(model(images), labels).backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == labels).sum().item() / len(y_pred)
            scheduler(epoch=epoch)
        comet_experiment_tracker_handler.log_metric(
            "train_loss_in_epoch", train_loss, step=i + 1, epoch=epoch + 1
        )

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss, train_acc


def gsam_single_epoch_training(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    scheduler: tuple,
    comet_experiment_tracker_handler: Any,
) -> tuple[float]:

    train_loss, train_acc = 0.0, 0.0

    for i, (X, y) in enumerate(train_data):
        images, labels = X.to(device), y.to(device)

        optimizer.set_closure(loss_function, images, labels)
        y_pred, loss = optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == labels).sum().item() / len(y_pred)
            scheduler[0].step()
            optimizer.update_rho_t()

        comet_experiment_tracker_handler.log_metric(
            "train_loss_in_epoch", train_loss, step=i + 1, epoch=epoch + 1
        )

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss, train_acc
