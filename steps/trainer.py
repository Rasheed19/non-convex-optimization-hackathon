import comet_ml

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models
from torch.optim import SGD

from utils.constants import NUM_CLASSES, IMAGE_DIMENSION, MODEL_DIR
from utils.optimizer import get_optimizer, enable_running_stats, disable_running_stats, StepLR, SAM, GSAM, CosineScheduler, ProportionScheduler

import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("API_KEY")
WORKSPACE = os.getenv("WORKSPACE")

exp = comet_ml.Experiment(project_name="hackathon", api_key=API_KEY, workspace=WORKSPACE)
exp_name = "exp_01"
exp.set_name(exp_name)


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


def training_base(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    train_loss: float, 
    train_acc: float, 
    epoch: int, 
) -> tuple[float]:
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

        exp.log_metric("train_loss_in_epoch", train_loss, step=i+1, epoch=epoch+1)


def training_SAM(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    train_loss: float, 
    train_acc: float,
    epoch: int,
    scheduler: StepLR,   
) -> tuple[float]:
    
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
        exp.log_metric("train_loss_in_epoch", train_loss, step=i+1, epoch=epoch+1)

def training_GSAM(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    train_loss: float, 
    train_acc: float,
    epoch: int,
    scheduler: list,  
) -> tuple[float]:
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
        exp.log_metric("train_loss_in_epoch", train_loss, step=i+1, epoch=epoch+1)

def single_epoch_training(
    model: nn.Module,
    train_data: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    scheduler: any, 
) -> tuple[float]:

    model.train()

    train_loss, train_acc = 0.0, 0.0

    if isinstance(optimizer, SAM):
        training_SAM(
            model = model,
            train_data = train_data,
            loss_function = loss_function,
            optimizer = optimizer,
            device = device,
            train_loss = train_loss, 
            train_acc = train_acc, 
            epoch=epoch,
            scheduler=scheduler, 
        )
        
    elif isinstance(optimizer, GSAM):
        training_GSAM(
            model = model,
            train_data = train_data,
            loss_function = loss_function,
            optimizer = optimizer,
            device = device,
            train_loss = train_loss, 
            train_acc = train_acc, 
            epoch=epoch,
            scheduler=scheduler, 
        )

    else:
        training_base(
            model = model,
            train_data = train_data,
            loss_function = loss_function,
            optimizer = optimizer,
            device = device,
            train_loss = train_loss, 
            train_acc = train_acc,
            epoch = epoch, 
        )


    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss, train_acc


def single_epoch_testing(
    model: nn.Module,
    test_data: DataLoader,
    loss_function: nn.Module,
    device: str,
    epoch: int
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

            exp.log_metric("test_loss_in_epoch", test_loss, step=i+1, epoch=epoch+1)

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

    model = create_model().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")
        model = nn.DataParallel(model)

    optimizer_params["params"] = model.parameters()

    if optimizer_name == 'gsam':
        scheduler_ = CosineScheduler(T_max=epochs*len(train_data), max_value=optimizer_params['lr'], min_value=0.0, optimizer=SGD)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler_, max_lr=optimizer_params['lr'], min_lr=0.0,
            max_value=optimizer_params['rho_max'], min_value=optimizer_params['rho_min'])
        scheduler = (scheduler_, rho_scheduler)
        base_optimizer = SGD(model.parameters(), lr=optimizer_params['lr'], momentum=optimizer_params['momentum'], weight_decay=optimizer_params['weight_decay'])
        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=optimizer_params['alpha'], rho_scheduler=scheduler[1], adaptive=optimizer_params['adaptive'])

    else:
        optimizer = get_optimizer(
            optimizer_name=optimizer_name, optimizer_params=optimizer_params
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    

    with exp.train():
        exp_params = {
            "neural_network": {
                "type": "mobilenet_v2(weights=None)",
                "batch_size": batch_size,
                "num_gpus": 3,
                "num_trainable_params": "3,600,000",
            },
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.001,
            },
            "loss_function": {
                "type": "nn.CrossEntropyLoss()",
            },
        }
        exp.log_parameters(parameters=exp_params)

    with exp.train():
        exp_params = {
            "neural_network": {
                "type": "mobilenet_v2(weights=None)",
                "batch_size": batch_size,
                "num_gpus": 3,
                "num_trainable_params": "3,600,000",
            },
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.001,
            },
            "loss_function": {
                "type": "nn.CrossEntropyLoss()",
            },
        }
        exp.log_parameters(parameters=exp_params)
    
    if isinstance(optimizer, SAM):
        scheduler = StepLR(optimizer=optimizer, learning_rate=optimizer_params['lr'], total_epochs=epochs)
    elif isinstance(optimizer, GSAM):
        pass
    else:
        scheduler = None    

    for epoch in range(epochs):
        train_loss, train_acc = single_epoch_training(
            model=model,
            train_data=train_data,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch, 
            scheduler=scheduler,
        )

        test_loss, test_acc = single_epoch_testing(
            model=model, test_data=test_data, loss_function=loss_function, device=device, epoch=epoch
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

        exp.log_metric("train_loss_end_epoch", train_loss, epoch=epoch+1)
        exp.log_metric("test_loss_end_epoch", test_loss, epoch=epoch+1)
        exp.log_metric("train_accuracy_end_epoch", train_acc, epoch=epoch+1)
        exp.log_metric("test_accuracy_end_epoch", test_acc, epoch=epoch+1)

    torch.save(
        obj=model.state_dict(),
        f=f"{MODEL_DIR}/model_batch_size={batch_size}_optimizer={optimizer_name}_epochs={epochs}.pt",
    )
    return model, history
