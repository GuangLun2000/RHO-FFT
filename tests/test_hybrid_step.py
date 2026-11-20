"""``hybrid_step`` on a toy split model: exact first-order part, forward-gradient zeroth-order part.

The toy wrapper implements the three methods ``hybrid_step`` uses, so the test needs no pretrained model.
"""
import inspect

import torch
import torch.nn.functional as F

from ho_fl.algorithms import hybrid_step, optimizer_for
from ho_fl.config import Config
from ho_fl.model import partition_parameters
from ho_fl.randomness import RoundDirections
from ho_fl.train import Backend

DEPTH, WIDTH, CLASSES, CUT = 4, 6, 3, 2


class ToyWrapper:
    """Residual blocks ``h <- h + tanh(h) @ W_i`` whose weights are named like the LoRA tensors of layer ``i``."""

    def __init__(self, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.randn(16, WIDTH, generator=generator)
        self.head = torch.randn(CLASSES, WIDTH, generator=generator)
        self.params = {f"toy.layers.{i}.lora.weight": torch.nn.Parameter(0.3 * torch.randn(WIDTH, WIDTH, generator=generator))
                       for i in range(DEPTH)}

    def block(self, index, hidden):
        return hidden + torch.tanh(hidden) @ self.params[f"toy.layers.{index}.lora.weight"]

    def forward_prefix(self, input_ids, attention_mask, cut):
        hidden = self.embedding[input_ids]
        for index in range(cut):
            hidden = self.block(index, hidden)
        return hidden, None

    def forward_suffix(self, hidden, context, cut, last_only=False, token_ids=None):
        for index in range(cut, DEPTH):
            hidden = self.block(index, hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        return hidden @ self.head.T

    def forward(self, input_ids, attention_mask, token_ids=None):
        hidden, context = self.forward_prefix(input_ids, attention_mask, 0)
        return self.forward_suffix(hidden, context, 0, last_only=token_ids is not None, token_ids=token_ids)


def make_batch():
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 16, (5, 7), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": torch.randint(0, CLASSES, (5,), generator=generator)}


def full_loss(wrapper, batch):
    return F.cross_entropy(wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=True)[:, -1].float(), batch["labels"])


def run_step(master_seed=99):
    cfg = Config().resolve()
    wrapper, batch = ToyWrapper(), make_batch()
    before = {name: param.detach().clone() for name, param in wrapper.params.items()}
    prefix, suffix = partition_parameters(wrapper.params, CUT)
    loss, scalars = hybrid_step(cfg, wrapper, wrapper.params, prefix, batch, None, optimizer_for(wrapper.params, cfg),
                                RoundDirections(master_seed), 0)
    return cfg, wrapper, batch, before, prefix, suffix, loss, scalars


def test_loss_and_first_order_gradients_are_exact():
    cfg, wrapper, batch, before, prefix, suffix, loss, scalars = run_step()
    reference = ToyWrapper()
    expected_loss = full_loss(reference, batch)
    expected = torch.autograd.grad(expected_loss, list(reference.params.values()))
    expected = dict(zip(reference.params, expected))
    assert torch.allclose(loss, expected_loss.detach(), rtol=1e-6, atol=0)
    for name in suffix:
        assert torch.allclose(wrapper.params[name].grad, expected[name], rtol=1e-5, atol=1e-7), name
    assert len(scalars) == cfg.q


def test_zeroth_order_gradient_is_the_forward_gradient_estimate():
    cfg, wrapper, batch, before, prefix, suffix, loss, scalars = run_step(master_seed=99)
    reference = ToyWrapper()
    ref_prefix, _ = partition_parameters(reference.params, CUT)
    with torch.no_grad():
        hidden, context = reference.forward_prefix(batch["input_ids"], batch["attention_mask"], CUT)
    h_cut = hidden.detach().requires_grad_(True)
    logits = reference.forward_suffix(h_cut, context, CUT, last_only=True)[:, -1].float()
    lam, = torch.autograd.grad(F.cross_entropy(logits, batch["labels"]), h_cut)
    true_gradient = dict(zip(reference.params, torch.autograd.grad(full_loss(reference, batch), list(reference.params.values()))))
    directions = RoundDirections(99)
    estimate = {name: torch.zeros_like(param) for name, param in ref_prefix.items()}
    for p in range(cfg.q):
        u = directions.get(ref_prefix, 0, p)
        with torch.no_grad():
            for name, param in ref_prefix.items():
                param.copy_(before[name]).add_(u[name], alpha=cfg.mu)
            h_plus, _ = reference.forward_prefix(batch["input_ids"], batch["attention_mask"], CUT)
            for name, param in ref_prefix.items():
                param.copy_(before[name])
        v = torch.sum(lam * (h_plus - hidden))
        assert torch.allclose(scalars[p], v, rtol=1e-4, atol=1e-9)
        # v / mu is the directional derivative of the loss along u, up to the finite-difference error.
        directional = sum(torch.sum(true_gradient[name] * u[name]) for name in ref_prefix)
        assert torch.allclose(v / cfg.mu, directional, rtol=0.05, atol=1e-3)
        for name in estimate:
            estimate[name] += u[name] * (v / (cfg.q * cfg.mu))
    for name in prefix:
        assert torch.allclose(wrapper.params[name].grad, estimate[name], rtol=1e-4, atol=1e-7), name


def test_step_starts_from_the_unperturbed_parameters():
    cfg, wrapper, batch, before, prefix, suffix, loss, scalars = run_step()
    for name, param in wrapper.params.items():
        change = (param.detach() - before[name]).abs().max().item()
        # One AdamW step moves every coordinate by about lr; a leftover perturbation would be about mu = 100 * lr.
        assert 0 < change < 2 * cfg.lr, (name, change)


def test_positional_signatures_used_by_other_entry_points_are_unchanged():
    """replay.py and profile.py call these positionally; new parameters must come after and have defaults."""
    expected = {hybrid_step: ["cfg", "wrapper", "params", "prefix", "batch", "tokens", "optimizer", "directions", "local_step"],
                optimizer_for: ["params", "cfg"],
                Backend.local_step: ["self", "cid", "indices", "optimizer", "directions", "e", "reference"],
                Backend.optimizer: ["self", "params"]}
    for function, names in expected.items():
        parameters = list(inspect.signature(function).parameters.values())
        assert [p.name for p in parameters[:len(names)]] == names, function
        assert all(p.default is not inspect.Parameter.empty for p in parameters[len(names):]), function
