# implement all optimizers her
import math
import numpy as np
from typing import Iterable
import torch
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer, required
import contextlib
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm
from torch_sgld import SGLD


def get_optimizer(optimizer_name: str, optimizer_params: dict) -> Optimizer:

    # FIXME: I think GSAM should be removed??
    optimizer_dict = dict(adam=Adam, sgd=SGD, sgld=SGLD, sam=SAM)

    return optimizer_dict[optimizer_name](**optimizer_params)


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


# class SGLD(Optimizer):
#     """
#     Implements SGLD algorithm based on
#         https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/src/langevin_sampling/SGLD.py
#     and
#         https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

#     Built on the PyTorch SGD implementation
#     (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
#     """

#     def __init__(
#         self,
#         params,
#         lr=required,
#         momentum=0,
#         dampening=0,
#         weight_decay=0,
#         nesterov=False,
#     ):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

#         defaults = dict(
#             lr=lr,
#             momentum=momentum,
#             dampening=dampening,
#             weight_decay=weight_decay,
#             nesterov=nesterov,
#         )
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
#         super(SGLD, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(SGLD, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault("nesterov", False)

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             weight_decay = group["weight_decay"]
#             momentum = group["momentum"]
#             dampening = group["dampening"]
#             nesterov = group["nesterov"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if weight_decay != 0:
#                     d_p.add_(p.data, alpha=weight_decay)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if "momentum_buffer" not in param_state:
#                         buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state["momentum_buffer"]
#                         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
#                     if nesterov:
#                         d_p = d_p.add(momentum, buf)
#                     else:
#                         d_p = buf

#                 p.data.add_(d_p, alpha=-group["lr"])
#                 noise_std = torch.tensor([2 * group["lr"]])
#                 noise_std = noise_std.sqrt()
#                 # noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std
#                 noise = torch.randn(p.data.size()) * noise_std
#                 p.data.add_(noise)

#         return 1.0


class SAM(Optimizer):
    """
    Implements SAM algorithm
        https://github.com/davda54/sam/blob/main/sam.py
    """

    def __init__(self, params, base_optimizer=SGD, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GSAM(Optimizer):
    """
    Implements GSAM algorithm
        https://github.com/juntang-zhuang/GSAM/blob/main/gsam/gsam.py
    """

    def __init__(
        self,
        params,
        model=None,
        gsam_alpha=0.01,
        rho_scheduler=None,
        base_optimizer=SGD,
        adaptive=False,
        perturb_eps=1e-12,
        grad_reduce="mean",
        **kwargs,
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == "mean":
            if hasattr(ReduceOp, "AVG"):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == "sum":
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p].keys():
                    p.data.sub_(self.state[p]["e_w"])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                inner_prod += torch.sum(self.state[p]["old_g"] * p.grad.data)

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by="old_g")

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                vertical = self.state[p][
                    "old_g"
                ] - cosine * old_grad_norm * p.grad.data / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.data.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack(
                    [
                        ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(
                            p=2
                        )
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2,
            )
        else:
            norm = torch.norm(
                torch.stack(
                    [
                        (
                            (torch.abs(p.data) if weight_adaptive else 1.0)
                            * self.state[p][by]
                        ).norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2,
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value


class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        assert (max_lr > min_lr) or ((max_lr == min_lr) and (max_value == min_value)), (
            "Current scheduler for `value` is scheduled to evolve proportionally to `lr`,"
            "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;"
            "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
        )

        assert max_value >= min_value

        self.step()  # take 1 step during initialization to get self._last_lr

    def lr(self):
        return self._last_lr[0]

    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]["lr"]

        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (
                lr - self.min_lr
            ) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value

        self._last_lr = [value]
        return value


class SchedulerBase:
    def __init__(
        self,
        T_max,
        max_value,
        min_value=0.0,
        init_value=0.0,
        warmup_steps=0,
        optimizer=None,
    ):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]

        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = (
                self.init_value
                + (self.max_value - self.init_value) * self.t / self.warmup_steps
            )
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = value

        self._last_lr = [value]
        return value

    def step_func(self):
        pass

    def lr(self):
        return self._last_lr[0]


class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (
            self.t - self.warmup_steps
        ) / (self.total_steps - self.warmup_steps)
        return value


class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (
            (self.t - self.warmup_steps)
            / (self.total_steps - self.warmup_steps)
            * math.pi
        )
        value = (
            self.min_value
            + (self.max_value - self.min_value) * (np.cos(phase) + 1.0) / 2.0
        )
        return value


class PolyScheduler(SchedulerBase):
    def __init__(self, poly_order=-0.5, *args, **kwargs):
        super(PolyScheduler, self).__init__(*args, **kwargs)
        self.poly_order = poly_order
        assert (
            poly_order <= 0
        ), "Please check poly_order<=0 so that the scheduler decreases with steps"

    def step_func(self):
        value = (
            self.min_value
            + (self.max_value - self.min_value)
            * (self.t - self.warmup_steps) ** self.poly_order
        )
        return value


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3 / 10:
            lr = self.base
        elif epoch < self.total_epochs * 6 / 10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8 / 10:
            lr = self.base * 0.2**2
        else:
            lr = self.base * 0.2**3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
