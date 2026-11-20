"""The current step functions against the frozen upstream ones (invariant I1 at the level of a local step)."""
import torch

import legacy_steps
from ho_fl import algorithms, classification
from ho_fl.config import Config
from ho_fl.model import partition_parameters
from ho_fl.randomness import RoundDirections
from ho_fl.train import Backend, logical_uplink
from test_hybrid_step import CUT, ToyWrapper, make_batch


def three_steps(module):
    cfg = Config().resolve()
    wrapper, batch = ToyWrapper(), make_batch()
    prefix, _ = partition_parameters(wrapper.params, CUT)
    optimizer = module.optimizer_for(wrapper.params, cfg)
    directions = RoundDirections(1234)
    trace = []
    for step in range(3):
        loss, scalars = module.hybrid_step(cfg, wrapper, wrapper.params, prefix, batch, None, optimizer, directions, step)
        trace.append((loss, scalars, {n: p.detach().clone() for n, p in wrapper.params.items()},
                      {n: p.grad.detach().clone() for n, p in wrapper.params.items()}))
    return trace, optimizer


def test_hybrid_step_is_bitwise_identical_to_upstream_over_consecutive_steps():
    expected, _ = three_steps(legacy_steps)
    actual, _ = three_steps(algorithms)
    for (loss_a, scalars_a, params_a, grads_a), (loss_b, scalars_b, params_b, grads_b) in zip(actual, expected):
        assert torch.equal(loss_a, loss_b) and len(scalars_a) == len(scalars_b) == 2
        assert all(torch.equal(a, b) for a, b in zip(scalars_a, scalars_b))
        assert all(torch.equal(params_a[n], params_b[n]) and torch.equal(grads_a[n], grads_b[n]) for n in params_b)


def test_optimizer_settings_are_those_of_upstream():
    _, expected = three_steps(legacy_steps)
    _, actual = three_steps(algorithms)
    assert type(actual) is type(expected) and actual.defaults == expected.defaults


def test_evaluation_is_identical_to_upstream(tiny_config):
    cfg = tiny_config("sst2")
    b = Backend(cfg)
    arguments = (b.wrapper, b.data["validation"], b.tokens, b.tokenizer.pad_token_id, cfg.eval_batch_size, cfg.device)
    assert classification.evaluate(*arguments) == legacy_steps.evaluate(*arguments)


def test_logical_uplink_is_identical_to_upstream():
    for method, bottom in (("hofl", 73728), ("hofl", 221184), ("fedavg", 0), ("fedprox", 0), ("decomfl", 294912)):
        cfg = Config(method=method).resolve()
        assert logical_uplink(cfg, 294912, bottom) == legacy_steps.logical_uplink(cfg, 294912, bottom)
