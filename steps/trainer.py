from torch import nn, save, load, hub
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from typing import Any
from torchvision import models

from utils.constants import NUM_CLASSES
from utils.optimizer import get_optimizer

import comet_ml
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("API_KEY")
WORKSPACE = os.getenv("WORKSPACE")

exp = comet_ml.Experiment(project_name="hackathon")
exp_name = "exp_01"
exp.set_name(exp_name)

exp_params = {
    "neural_network": {
        "type": "squeezenet1_0(weights=None)",
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


class DeepNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,
        #         out_channels=8,
        #         kernel_size=3,
        #         padding=1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=8,
        #         out_channels=16,
        #         kernel_size=3,
        #     ),
        #     nn.Conv2d(
        #         in_channels=16,
        #         out_channels=32,
        #         kernel_size=3,
        #     ),
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=3,
        #     ),
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=128,
        #         kernel_size=3,
        #     ),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Linear(128 * 7 * 7, 512),
        #     nn.Linear(512, NUM_CLASSES),
        # )
        self.model = nn.Sequential(
            nn.Linear(3 * 224 * 224, 1024),
            nn.Linear(1024, NUM_CLASSES),
        )

    def forward(self, x):
        return self.model(x)


def model_trainer(
    training_data: DataLoader,
    optimizer_name: str,
    optimizer_params: dict,
    epochs: int,
    device: str,
    loss_function: Any = nn.CrossEntropyLoss(),
) -> nn.Module:

    # classifier = nn.Sequential(
    #     nn.Linear(1000, 1024),
    #     nn.LeakyReLU(),
    #     nn.Linear(1024, NUM_CLASSES),
    # )
    model = models.squeezenet1_0(weights=None)  # DeepNN().to(device=device)
    # model.classifier = classifier

    print(summary(model, (3, 224, 224)))

    # print(model)
    # print(len(model))
    # optimization set up
    optimizer_params["params"] = model.parameters()
    optimizer = get_optimizer(
        optimizer_name=optimizer_name, optimizer_params=optimizer_params
    )

    for epoch in range(epochs):  # train for 10 epochs
        for i, batch in enumerate(training_data):
            X, y = batch
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = loss_function(yhat, y)

            # apply back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"At epoch: {epoch + 1} and batch: {i + 1}")

        exp.log_metric("loss", step=epoch+1, epoch=epoch+1)
        print(f"Epoch: {epoch + 1}, loss: {loss.item()}")

    #     # Calculate and accumulate accuracy metric across all batches
    #     y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    #     train_acc += (y_pred_class == labels).sum().item()/len(y_pred)

    # # Adjust metrics to get average loss and accuracy per batch
    # train_loss = train_loss / len(dataloader)
    # train_acc = train_acc / len(dataloader)
    # return train_loss, train_acc

    # with open(f"{Definition.ROOT_DIR}/models/model_state.pt", "wb") as f:
    #     save(clf.state_dict(), f)
