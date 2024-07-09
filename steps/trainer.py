from torch import nn, save, load, hub
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from typing import Any

from utils.constants import NUM_CLASSES
from utils.optimizer import get_optimizer


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
        nn.Sequential(
            nn.Linear(1920, 1024),
            nn.LeakyReLU(),
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

    model = DeepNN().to(device=device)

    # optimization set up
    optimizer_params["params"] = model.parameters()
    optimizer = get_optimizer(
        optimizer_name=optimizer_name, optimizer_params=optimizer_params
    )

    for epoch in range(epochs):  # train for 10 epochs
        for batch in training_data:
            X, y = batch
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = loss_function(yhat, y)

            # apply back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, loss: {loss.item()}")

    #     # Calculate and accumulate accuracy metric across all batches
    #     y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    #     train_acc += (y_pred_class == labels).sum().item()/len(y_pred)

    # # Adjust metrics to get average loss and accuracy per batch
    # train_loss = train_loss / len(dataloader)
    # train_acc = train_acc / len(dataloader)
    # return train_loss, train_acc

    # with open(f"{Definition.ROOT_DIR}/models/model_state.pt", "wb") as f:
    #     save(clf.state_dict(), f)
