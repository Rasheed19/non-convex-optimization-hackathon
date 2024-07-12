import random
import torch
import numpy as np
from steps.trainer import create_model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score(accu):
    return 100 * sigmoid(0.1 * (accu - 50)) + 100 * (1 - sigmoid(5))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(filepath, device):
    model = create_model().to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(filepath))

    return model


def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
