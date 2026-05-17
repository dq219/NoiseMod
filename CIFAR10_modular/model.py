"""ResNet-18 adapted to film-roll CIFAR-10 with N concatenated frames."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ModularResNet18(nn.Module):
    """ResNet-18 that emits one length-``num_classes`` logit vector per frame.

    The stem is replaced with a 3x3 stride-1 convolution and the initial
    maxpool is dropped — the standard CIFAR ResNet adaptation — so the
    spatial resolution after ``layer4`` is ``(4, 4 * N)`` for a
    ``(3, 32, 32 * N)`` input. An adaptive average pool collapses the height
    to 1 while preserving N positions along the width, giving a flattened
    feature of size ``512 * N``. A two-layer MLP head then maps
    ``512 * N`` → ``512 * N`` (with ReLU) → ``num_classes * N`` outputs which
    are reshaped to ``(B, N, num_classes)``. The hidden layer gives the
    weighted-activity regularizer something non-trivial to push on: with
    only one FC layer the irrelevant inputs trivially get small outgoing
    weights, but here the regularizer drives the *hidden* units to
    specialise.

    Args:
        N: number of frames concatenated in the input.
        num_classes: number of classes per frame (10 for CIFAR-10).
    """

    def __init__(self, N: int, num_classes: int = 10, hidden_dim: int | None = None):
        super().__init__()
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = N
        self.num_classes = num_classes
        self.hidden_dim = 512 * N if hidden_dim is None else int(hidden_dim)

        backbone = torchvision.models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, N))
        self.fc1 = nn.Linear(512 * N, self.hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes * N)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional trunk → flattened pre-MLP features, shape ``(B, 512 * N)``."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Through fc1 + ReLU. Shape ``(B, hidden_dim)``."""
        return self.act(self.fc1(self._forward_features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._forward_hidden(x)
        logits = self.fc2(hidden)
        return logits.view(hidden.size(0), self.N, self.num_classes)

    def weighted_act_regularizer(self, hidden: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        """Weighted-activity regularizer on the MLP hidden layer.

        Mirrors ``regularizer_WeightedAct`` in
        ``modularisation_via_noise/utils/network.py``: each hidden unit's
        activity is weighted by the squared magnitudes of its outgoing
        connections in ``self.fc2``, summed over fc2's outputs and averaged
        over the batch:

            E_batch[ sum_j ( |h|^alpha @ (W2.T)^2 )_j ]

        where ``W2 = self.fc2.weight``. Applied only to the hidden layer,
        so the regularizer's pressure is to make ``fc2`` route each hidden
        unit to outputs of a single frame — driving modularity.

        Args:
            hidden: post-ReLU hidden activation, shape ``(B, hidden_dim)``.
            alpha: exponent on the absolute activation. Defaults to 2.0.
        """
        pass_error = (torch.abs(hidden) ** alpha) @ (self.fc2.weight.t() ** 2)
        return torch.mean(torch.sum(pass_error, dim=1))

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        alpha: float = 2.0,
        reg_weight: float = 0.0,
    ) -> dict:
        """Run one training step on a single batch.

        Named ``train_step`` rather than ``train`` to avoid shadowing
        ``nn.Module.train``.

        Args:
            x: input batch, shape ``(B, 3, 32, 32 * N)`` (the convention
                produced by :class:`data.ConcatCIFAR10`).
            y: per-frame labels, shape ``(B, N)``, dtype long.
            optimizer: a torch optimizer already bound to this model's
                parameters. ``zero_grad`` and ``step`` are called inside.
            alpha: exponent passed to :meth:`weighted_act_regularizer`.
            reg_weight: scalar weight on the regularizer. Set to 0 to
                disable.

        Returns:
            Dict with python-float values: ``loss`` (total), ``ce``
            (cross-entropy), ``reg`` (raw regularizer value, before
            ``reg_weight``).
        """
        self.train()
        optimizer.zero_grad(set_to_none=True)

        hidden = self._forward_hidden(x)
        logits = self.fc2(hidden).view(x.size(0), self.N, self.num_classes)

        ce = F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            y.reshape(-1),
        )
        reg = self.weighted_act_regularizer(hidden, alpha=alpha)
        loss = ce + reg_weight * reg

        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "ce": ce.item(), "reg": reg.item()}
