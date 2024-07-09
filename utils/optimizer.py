# implement all optimizers her
from torch.optim import Adam, SGD, Optimizer


def get_optimizer(optimizer_name: str, optimizer_params: dict) -> Optimizer:

    optimizer_dict = dict(adam=Adam, sgd=SGD)

    return optimizer_dict[optimizer_name](**optimizer_params)
