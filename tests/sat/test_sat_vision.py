"""The vision model of the satellite scenario: ViT with LoRA and a trainable head (brief section 7.4).

The tests use the random 12-block ViT of ``ho_fl.sat.testing``. Gradients are compared with a float32 backbone,
where autograd is an exact reference; the bfloat16 backbone of real runs gets a smoke test.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ho_fl import model as upstream_model
from ho_fl.algorithms import hybrid_step, optimizer_for
from ho_fl.io import tensor_hash
from ho_fl.model import block_index, partition_parameters
from ho_fl.randomness import RoundDirections
from ho_fl.sat.config import SatConfig
from ho_fl.sat.testing import TINY_IMAGE_SIZE, build_tiny_assets
from ho_fl.sat.vision import ViTPartition, image_batch, load_vit

DEPTH, CLASSES = 12, 10


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return build_tiny_assets(tmp_path_factory.mktemp("sat-vision"))


def load(assets, dtype=torch.float32, seed=42):
    cfg = SatConfig(**{**assets["t2"], "seed": seed}).resolve().train_config()
    peft, wrapper, params = load_vit(cfg, CLASSES, DEPTH, dtype=dtype)
    return cfg, peft, wrapper, params


def make_batch(count=6, seed=3):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (count, 8, 8, 3), generator=generator, dtype=torch.uint8)
    labels = torch.randint(0, CLASSES, (count,), generator=generator)
    return image_batch(images, labels, list(range(count)), TINY_IMAGE_SIZE, "cpu")


def shake(params, seed=11):
    """Move every trainable tensor off its initial value (LoRA B starts at zero, which hides the gradient of A)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in params.values():
            param.add_(0.05 * torch.randn(param.shape, generator=generator))


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def test_trainable_tensors_are_lora_and_the_head(assets):
    cfg, peft, wrapper, params = load(assets, dtype=torch.bfloat16)
    head = [name for name in params if "lora_" not in name]
    lora = [name for name in params if "lora_" in name]
    assert len(lora) == DEPTH * 2 * 2 and len(head) == 2 and all("classifier" in name for name in head)
    assert all(part in name for name in lora for part in ("vit.encoder.layer.",)) and {"query", "value"} == {name.split(".")[-4] for name in lora}
    assert all(param.dtype == torch.float32 for param in params.values())
    assert all(param.dtype == torch.bfloat16 for param in peft.parameters() if not param.requires_grad)
    assert [upstream_model.HEAD_BLOCK[name] for name in head] == [DEPTH, DEPTH]
    assert sorted({block_index(name) for name in lora}) == list(range(DEPTH)) and {block_index(name) for name in head} == {DEPTH}
    for cut in (0, 3, 6, 9, 12):
        bottom, top = partition_parameters(params, cut)
        assert len(bottom) == 4 * cut and set(head) <= set(top)
    assert isinstance(wrapper, ViTPartition) and len(wrapper.layers) == DEPTH
    # Sizes of the real model (appendix A): rank-8 LoRA on query and value of 12 blocks of width 768, a 10-class head.
    block, classifier = 2 * 2 * 8 * 768, 768 * 10 + 10
    assert (DEPTH * block, classifier, DEPTH * block + classifier) == (294912, 7690, 302602)
    assert [round(4 * ((DEPTH - cut) * block + classifier) / 1e6, 3) for cut in (0, 3, 6, 9, 12)] == [1.210, 0.915, 0.621, 0.326, 0.031]


def test_the_seed_fixes_the_head_and_the_lora_matrices(assets):
    first, second, other = (load(assets, seed=seed)[3] for seed in (42, 42, 43))
    assert tensor_hash(first) == tensor_hash(second) != tensor_hash(other)
    head = next(name for name in first if name.endswith("classifier.modules_to_save.default.weight"))
    assert float(first[head].abs().max()) > 0 and not torch.equal(first[head], other[head])
    assert all(float(param.abs().max()) == 0 for name, param in first.items() if "lora_B" in name)


def test_partition_reproduces_the_forward_pass_of_the_model(assets):
    cfg, peft, wrapper, params = load(assets)
    shake(params)
    batch = make_batch()
    with torch.no_grad():
        reference = peft(pixel_values=batch["input_ids"]).logits
        full = wrapper.forward(batch["input_ids"], batch["attention_mask"])
        assert full.shape == (6, 1, CLASSES) and torch.allclose(full[:, 0], reference, rtol=1e-5, atol=1e-6)
        for cut in (0, 3, 6, 9, 12):
            hidden, context = wrapper.forward_prefix(batch["input_ids"], batch["attention_mask"], cut)
            assert context is None and hidden.shape == (6, 1 + (TINY_IMAGE_SIZE // 8) ** 2, 32)
            assert torch.allclose(wrapper.forward_suffix(hidden, context, cut, last_only=True), full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cut", [0, 3, 6, 9, 12])
def test_first_order_gradients_match_autograd(cut, assets):
    cfg, peft, wrapper, params = load(assets)
    shake(params)
    batch = make_batch()
    loss = F.cross_entropy(wrapper.forward(batch["input_ids"], batch["attention_mask"])[:, -1].float(), batch["labels"])
    exact = dict(zip(params, torch.autograd.grad(loss, list(params.values()))))
    assert all(float(gradient.abs().max()) > 0 for gradient in exact.values())
    prefix, suffix = partition_parameters(params, cut)
    directions = RoundDirections(8)
    # A small radius, so that the finite difference of this float32 model is close to the directional derivative.
    small = SimpleNamespace(q=cfg.q, mu=1e-4)
    step_loss, scalars = hybrid_step(small, wrapper, params, prefix, batch, None, NoStep(), directions, 0, cut=cut)
    assert torch.allclose(step_loss, loss.detach(), rtol=1e-6, atol=0)
    for name in suffix:  # blocks at or above the cut and the head: backpropagation
        assert torch.allclose(params[name].grad, exact[name], rtol=1e-4, atol=1e-7), name
    assert len(scalars) == (small.q if cut else 0)
    for p, scalar in enumerate(scalars):  # blocks below the cut: v / mu is the directional derivative along u
        u = directions.get(prefix, 0, p)
        directional = sum(torch.sum(exact[name] * u[name]) for name in prefix)
        assert torch.allclose(scalar / small.mu, directional, rtol=0.05, atol=1e-3)
    for name in prefix:
        expected = sum(directions.get(prefix, 0, p)[name] * (scalars[p] / (small.q * small.mu)) for p in range(small.q))
        assert torch.allclose(params[name].grad, expected, rtol=1e-5, atol=1e-8), name


def test_a_step_with_the_bfloat16_backbone_is_finite_and_moves_every_tensor(assets):
    cfg, peft, wrapper, params = load(assets, dtype=torch.bfloat16)
    before = {name: param.detach().clone() for name, param in params.items()}
    prefix, _ = partition_parameters(params, 6)
    batch = make_batch()
    optimizer = optimizer_for(params, cfg)
    for step in range(2):  # LoRA A only moves once B is non-zero, i.e. from the second step on
        loss, scalars = hybrid_step(cfg, wrapper, params, prefix, batch, None, optimizer, RoundDirections(8), step, cut=6)
        assert torch.isfinite(loss) and len(scalars) == cfg.q and all(torch.isfinite(scalar) for scalar in scalars)
    assert all(torch.isfinite(param).all() and not torch.equal(param, before[name]) for name, param in params.items())


def test_image_batch_resizes_and_normalises():
    images = torch.zeros(4, 8, 8, 3, dtype=torch.uint8)
    images[1] = 255
    images[2, :, :4] = 255
    labels = torch.tensor([3, 1, 4, 1])
    batch = image_batch(images, labels, [2, 1, 0], 16, "cpu")
    assert batch["input_ids"].shape == (3, 3, 16, 16) and batch["attention_mask"] is None and batch["labels"].tolist() == [4, 1, 3]
    assert torch.all(batch["input_ids"][1] == 1.0) and torch.all(batch["input_ids"][2] == -1.0)
    assert batch["input_ids"][0].min() == -1.0 and batch["input_ids"][0].max() == 1.0
    assert batch["input_ids"][0, :, :, :6].eq(1.0).all() and batch["input_ids"][0, :, :, 10:].eq(-1.0).all()
    same = image_batch(images, labels, [1], 8, "cpu")["input_ids"]
    assert same.shape == (1, 3, 8, 8) and torch.all(same == 1.0)
