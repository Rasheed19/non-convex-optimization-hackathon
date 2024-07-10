import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models

from utils.constants import NUM_CLASSES, IMAGE_DIMENSION, MODEL_DIR
from utils.optimizer import get_optimizer


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
) -> tuple[float]:

    model.train()

    train_loss, train_acc = 0.0, 0.0

    for X, y in train_data:
        images, labels = X.to(device), y.to(device)

        y_pred = model(images)

        loss = loss_function(y_pred, labels)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(y_pred)

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss, train_acc


def single_epoch_testing(
    model: nn.Module,
    test_data: DataLoader,
    loss_function: nn.Module,
    device: str,
) -> tuple[float]:

    model.eval()

    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in test_data:
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
    loss_function: nn.Module = nn.CrossEntropyLoss(),
) -> tuple[nn.Module, dict]:

    model = create_model()
    optimizer_params["params"] = model.parameters()
    optimizer = get_optimizer(
        optimizer_name=optimizer_name, optimizer_params=optimizer_params
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):

        train_loss, train_acc = single_epoch_training(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = single_epoch_testing(
            model=model, test_data=test_data, loss_function=loss_function, device=device
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

    torch.save(
        obj=model.state_dict(),
        f=f"{MODEL_DIR}/model_batch_size={batch_size}_optimizer={optimizer_name}_epochs={epochs}.pt",
    )
    return model, history
