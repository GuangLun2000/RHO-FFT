"""The changes to upstream modules made for the satellite scenario (brief v2, section 4.4).

Each change comes with the check that the default behaviour is the upstream one (invariant I1): the explicit cut
against the frozen upstream step, the layer index of the names upstream knows, the configuration defaults.
"""
import tomllib
from pathlib import Path

import pytest
import torch

import legacy_steps
import ho_fl.train as train
from ho_fl import algorithms, model
from ho_fl.batches import collate
from ho_fl.config import Config
from ho_fl.model import block_index, partition_parameters, register_head
from ho_fl.randomness import RoundDirections
from ho_fl.train import Backend
from test_hybrid_step import DEPTH, ToyWrapper, full_loss, make_batch

REPO = Path(__file__).resolve().parents[2]


def toy_steps(function, cut, explicit, count=3):
    """``count`` consecutive steps on the toy model; ``explicit`` passes the cut instead of letting it be derived."""
    cfg = Config().resolve()
    wrapper, batch = ToyWrapper(), make_batch()
    prefix, _ = partition_parameters(wrapper.params, cut)
    optimizer = algorithms.optimizer_for(wrapper.params, cfg)
    directions = RoundDirections(1234)
    trace = []
    for step in range(count):
        extra = {"cut": cut} if explicit else {}
        loss, scalars = function(cfg, wrapper, wrapper.params, prefix, batch, None, optimizer, directions, step, **extra)
        trace.append((loss, scalars, {n: p.detach().clone() for n, p in wrapper.params.items()},
                      {n: p.grad.detach().clone() for n, p in wrapper.params.items()}))
    return trace


def assert_same_trace(actual, expected):
    assert len(actual) == len(expected)
    for (loss_a, scalars_a, params_a, grads_a), (loss_b, scalars_b, params_b, grads_b) in zip(actual, expected):
        assert torch.equal(loss_a, loss_b) and len(scalars_a) == len(scalars_b)
        assert all(torch.equal(a, b) for a, b in zip(scalars_a, scalars_b))
        assert all(torch.equal(params_a[n], params_b[n]) and torch.equal(grads_a[n], grads_b[n]) for n in params_b)


# --- hybrid_step: explicit cut ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("cut", [1, 2, 3])
def test_explicit_cut_is_bitwise_the_upstream_step_on_the_toy_model(cut):
    upstream = toy_steps(legacy_steps.hybrid_step, cut, explicit=False)
    assert_same_trace(toy_steps(algorithms.hybrid_step, cut, explicit=False), upstream)
    assert_same_trace(toy_steps(algorithms.hybrid_step, cut, explicit=True), upstream)


@pytest.mark.parametrize("cut", [3, 6, 9])
def test_explicit_cut_is_bitwise_the_upstream_step_on_the_tiny_model(cut, tiny_config):
    """The tier cuts of the 12-block tiny model stand for 8, 16 and 24 of Qwen (the GPU self-check uses those)."""
    cfg = tiny_config("sst2")
    b = Backend(cfg)
    state = {name: param.detach().clone() for name, param in b.params.items()}
    prefix = b.bottoms[cut]
    batch = collate(b.data["train"], list(range(cfg.batch_size)), b.tokenizer.pad_token_id, cfg.device)

    def two_steps(function, **extra):
        algorithms.restore(b.params, state)
        optimizer, directions, trace = b.optimizer(), RoundDirections(77), []
        for step in range(2):
            loss, scalars = function(cfg, b.wrapper, b.params, prefix, batch, b.tokens, optimizer, directions, step, **extra)
            trace.append((loss, scalars, {n: p.detach().clone() for n, p in b.params.items()},
                          {n: p.grad.detach().clone() for n, p in b.params.items()}))
        return trace

    upstream = two_steps(legacy_steps.hybrid_step)
    assert_same_trace(two_steps(algorithms.hybrid_step), upstream)
    assert_same_trace(two_steps(algorithms.hybrid_step, cut=cut), upstream)


def test_local_step_hands_the_cut_to_hybrid_step(tiny_config, monkeypatch):
    b = Backend(tiny_config("sst2"))
    seen = []

    def spy(cfg, wrapper, params, prefix, *rest, **keywords):
        seen.append((keywords["cut"], set(prefix)))
        return torch.zeros(()), []

    monkeypatch.setattr(train, "hybrid_step", spy)
    cid = next(index for index, cut in enumerate(b.cuts) if cut == 6)
    indices = list(range(b.cfg.batch_size))
    b.local_step(cid, indices, None, None, 0, None)
    b.local_step(cid, indices, None, None, 0, None, cut=9)
    assert seen == [(6, set(b.bottoms[6])), (9, set(b.bottoms[9]))]


# --- hybrid_step: cut 0 and cut = depth ------------------------------------------------------------------------------

def test_cut_zero_is_plain_backpropagation_and_draws_no_direction():
    cfg = Config().resolve()
    wrapper, batch = ToyWrapper(), make_batch()
    before = {name: param.detach().clone() for name, param in wrapper.params.items()}
    directions = RoundDirections(5)
    loss, scalars = algorithms.hybrid_step(cfg, wrapper, wrapper.params, {}, batch, None,
                                           algorithms.optimizer_for(wrapper.params, cfg), directions, 0, cut=0)
    reference = ToyWrapper()
    expected_loss = full_loss(reference, batch)
    expected = dict(zip(reference.params, torch.autograd.grad(expected_loss, list(reference.params.values()))))
    assert scalars == [] and directions.used_seeds == {} and directions.cache == {}
    assert torch.allclose(loss, expected_loss.detach(), rtol=1e-6, atol=0)
    for name, param in wrapper.params.items():
        assert torch.allclose(param.grad, expected[name], rtol=1e-6, atol=1e-6), name
        assert not torch.equal(param, before[name])


def test_cut_zero_matches_the_fedavg_gradient_on_the_tiny_model(tiny_config):
    cfg = tiny_config("sst2")
    b = Backend(cfg)
    batch = collate(b.data["train"], list(range(cfg.batch_size)), b.tokenizer.pad_token_id, cfg.device)
    algorithms.task_loss(b.wrapper, batch, b.tokens).backward()
    expected = {name: param.grad.detach().clone() for name, param in b.params.items()}
    for param in b.params.values():
        param.grad = None

    class NoStep:
        def zero_grad(self, set_to_none=True):
            pass

        def step(self):
            pass

    loss, scalars = algorithms.hybrid_step(cfg, b.wrapper, b.params, {}, batch, b.tokens, NoStep(), RoundDirections(5), 0, cut=0)
    assert scalars == [] and torch.isfinite(loss)
    assert all(expected[name].abs().max() > 0 for name in expected if "lora_B" in name)
    for name, param in b.params.items():
        assert torch.allclose(param.grad, expected[name], rtol=1e-6, atol=1e-6), name


def test_an_empty_prefix_without_a_cut_still_fails_as_upstream():
    cfg = Config().resolve()
    wrapper = ToyWrapper()
    with pytest.raises(ValueError):
        algorithms.hybrid_step(cfg, wrapper, wrapper.params, {}, make_batch(), None,
                               algorithms.optimizer_for(wrapper.params, cfg), RoundDirections(5), 0)


HEAD = "toy.classifier.weight"


class HeadedToy(ToyWrapper):
    """The toy model with a trainable head, named like a tensor outside the blocks (a ViT classifier)."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.params[HEAD] = torch.nn.Parameter(self.head.clone())

    def forward_suffix(self, hidden, context, cut, last_only=False, token_ids=None):
        for index in range(cut, DEPTH):
            hidden = self.block(index, hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        return hidden @ self.params[HEAD].T


@pytest.fixture
def toy_head(monkeypatch):
    monkeypatch.setitem(model.HEAD_BLOCK, HEAD, DEPTH)


@pytest.mark.parametrize("explicit", [True, False])
def test_cut_at_depth_trains_every_block_forward_only_and_the_head_by_backpropagation(explicit, toy_head):
    cfg = Config().resolve()
    wrapper, batch = HeadedToy(), make_batch()
    before = {name: param.detach().clone() for name, param in wrapper.params.items()}
    prefix, suffix = partition_parameters(wrapper.params, DEPTH)
    assert list(suffix) == [HEAD] and len(prefix) == DEPTH
    loss, scalars = algorithms.hybrid_step(cfg, wrapper, wrapper.params, prefix, batch, None,
                                           algorithms.optimizer_for(wrapper.params, cfg), RoundDirections(99), 0,
                                           **({"cut": DEPTH} if explicit else {}))
    reference = HeadedToy()
    ref_prefix, _ = partition_parameters(reference.params, DEPTH)
    expected_loss = full_loss(reference, batch)
    head_gradient, = torch.autograd.grad(expected_loss, [reference.params[HEAD]])
    assert torch.allclose(loss, expected_loss.detach(), rtol=1e-6, atol=0) and len(scalars) == cfg.q
    assert torch.allclose(wrapper.params[HEAD].grad, head_gradient, rtol=1e-5, atol=1e-7)
    # The blocks carry the forward-gradient estimate built from the activations after the last block.
    with torch.no_grad():
        hidden, context = reference.forward_prefix(batch["input_ids"], batch["attention_mask"], DEPTH)
    h_cut = hidden.detach().requires_grad_(True)
    logits = reference.forward_suffix(h_cut, context, DEPTH, last_only=True)[:, -1].float()
    lam, = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, batch["labels"]), h_cut)
    directions = RoundDirections(99)
    estimate = {name: torch.zeros_like(param) for name, param in ref_prefix.items()}
    for p in range(cfg.q):
        u = directions.get(ref_prefix, 0, p)
        with torch.no_grad():
            for name, param in ref_prefix.items():
                param.copy_(before[name]).add_(u[name], alpha=cfg.mu)
            h_plus, _ = reference.forward_prefix(batch["input_ids"], batch["attention_mask"], DEPTH)
            for name, param in ref_prefix.items():
                param.copy_(before[name])
        v = torch.sum(lam * (h_plus - hidden))
        assert torch.allclose(scalars[p], v, rtol=1e-4, atol=1e-9)
        for name in estimate:
            estimate[name] += u[name] * (v / (cfg.q * cfg.mu))
    for name in prefix:
        assert torch.allclose(wrapper.params[name].grad, estimate[name], rtol=1e-4, atol=1e-7), name
    assert all(not torch.equal(param, before[name]) for name, param in wrapper.params.items())


# --- block_index -----------------------------------------------------------------------------------------------------

def test_block_index_keeps_the_upstream_names_and_reads_vit_names(monkeypatch):
    monkeypatch.setattr(model, "HEAD_BLOCK", {})  # the registry is per process: a ViT loaded by another test fills it
    assert block_index("base_model.model.model.layers.27.self_attn.q_proj.lora_A.default.weight") == 27
    assert block_index("base_model.model.model.decoder.layers.11.self_attn.v_proj.lora_B.default.weight") == 11
    assert block_index("base_model.model.vit.encoder.layer.3.attention.attention.query.lora_A.default.weight") == 3
    assert block_index("base_model.model.vit.encoder.layer.11.attention.attention.value.lora_B.default.weight") == 11
    for name in ("lm_head.weight", "base_model.model.classifier.modules_to_save.default.weight", "x.layerwise.3.w"):
        with pytest.raises(AssertionError):
            block_index(name)


def test_a_registered_head_is_block_depth_and_always_in_the_suffix(monkeypatch):
    monkeypatch.setattr(model, "HEAD_BLOCK", {})
    head = ["base_model.model.classifier.modules_to_save.default.bias", "base_model.model.classifier.modules_to_save.default.weight"]
    blocks = [f"base_model.model.vit.encoder.layer.{i}.attention.attention.{proj}.lora_{ab}.default.weight"
              for i in range(12) for proj in ("query", "value") for ab in "AB"]
    register_head(head, 12)
    assert model.HEAD_BLOCK == {name: 12 for name in head} and [block_index(name) for name in head] == [12, 12]
    params = {name: torch.zeros(1) for name in sorted(head + blocks)}
    for cut in (0, 3, 6, 9, 12):
        bottom, top = partition_parameters(params, cut)
        assert set(head) <= set(top) and len(bottom) == 4 * cut and len(top) == 4 * (12 - cut) + 2
    monkeypatch.setattr(model, "HEAD_BLOCK", {})
    with pytest.raises(AssertionError):
        block_index(head[0])


# --- Config.scenario and the entry points ----------------------------------------------------------------------------

def test_scenario_lifts_only_the_thirty_device_requirement():
    assert Config().scenario == "" and Config().resolve().n == 30
    with pytest.raises(ValueError):
        Config(n=120, k=18).resolve()
    cfg = Config(n=120, k=18, scenario="s1").resolve()
    assert (cfg.n, cfg.k, cfg.scenario) == (120, 18, "s1")
    for fields in ({"k": 0}, {"k": 121}, {"E": 0}):
        with pytest.raises(ValueError):
            Config(n=120, scenario="s1", **fields).resolve()
    # The label of a run and its aggregation rule do not depend on the scenario.
    assert Config(scenario="s1").resolve().method_label == Config().resolve().method_label


def test_entry_points_and_the_optional_dependency_group_are_declared():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    scripts = project["scripts"]
    from ho_fl.sat import cli

    for command in ("trace", "train", "tune", "topology", "figures"):
        module, function = scripts[f"ho-fl-sat-{command}"].split(":")
        assert module == "ho_fl.sat.cli" and callable(getattr(cli, function))
        assert cli.COMMANDS[command] == function
    for name, target in (("ho-fl-train", "ho_fl.train:main"), ("ho-fl-profile", "ho_fl.profile:main"),
                         ("ho-fl-memory", "ho_fl.memory:main"), ("ho-fl-replay", "ho_fl.replay:main")):
        assert scripts[name] == target
    assert "global-land-mask==1.0.0" in project["optional-dependencies"]["sat"]
