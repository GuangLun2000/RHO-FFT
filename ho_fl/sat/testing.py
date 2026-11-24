"""Synthetic assets for CPU tests and dry runs of the satellite scenario.

A random tiny language model (from ``ho_fl.testing``), a random tiny ViT and small synthetic stand-ins for HumAID
and EuroSAT, written where the real loaders keep their caches. With ``SatConfig(tiny=True)`` a run uses them and
goes through the same code as a real run in seconds, without a network or a GPU. Both models have 12 blocks, so
the cut table is the one of the vision task. Numbers produced with these assets say nothing about the method.
"""
from pathlib import Path

import numpy as np
import torch

from ..io import write_json
from ..testing import FIRST_WORD_ID, VOCAB_SIZE, build_tiny_model
from .datasets import SPLITS, data_dir, humaid_hash, write_eurosat

TINY_COUNTS = {"train": 2520, "validation": 240, "test": 360}   # 21 training examples per satellite
TINY_TEXT_CLASSES = 4       # the tiny tokenizer knows the verbalizers A to D
TINY_IMAGE_CLASSES = 10
TINY_EVENTS = 19
TINY_IMAGE_SIZE = 16


def build_tiny_vit(directory, seed=0):
    """Save a randomly initialised 12-block ViT backbone to ``directory`` (loadable through ``model_path``).

    Like the ImageNet-21k checkpoint of the real task it has no classification head, so loading it as a
    classifier initialises the head from the seed set by the loader.
    """
    from transformers import ViTConfig, ViTModel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = ViTConfig(hidden_size=32, num_hidden_layers=12, num_attention_heads=4, intermediate_size=64,
                       image_size=TINY_IMAGE_SIZE, patch_size=8, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        ViTModel(config, add_pooling_layer=False).save_pretrained(directory)
    finally:
        torch.random.set_rng_state(state)
    return directory


def build_tiny_humaid(cache_dir, seed=0):
    """A stand-in for HumAID: 19 events, 4 classes, texts made of the tiny vocabulary. The words of a text depend
    on its class and the class mix depends on the event, so there is something to learn and planes differ."""
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(FIRST_WORD_ID, VOCAB_SIZE)]
    band = len(words) // TINY_TEXT_CLASSES
    mix = rng.dirichlet(np.full(TINY_TEXT_CLASSES, 0.8), size=TINY_EVENTS)
    data = {"classes": [f"class_{c}" for c in range(TINY_TEXT_CLASSES)], "events": [f"event_{e:02d}" for e in range(TINY_EVENTS)]}
    for split in SPLITS:
        events = rng.integers(0, TINY_EVENTS, size=TINY_COUNTS[split])
        labels = np.array([rng.choice(TINY_TEXT_CLASSES, p=mix[event]) for event in events])
        texts = []
        for label in labels:
            length = int(rng.integers(4, 9))
            own = rng.integers(label * band, (label + 1) * band, size=length)
            noise = rng.integers(0, len(words), size=length)
            texts.append(" ".join(words[o if keep else n] for o, n, keep in zip(own, noise, rng.random(length) < 0.8)))
        data[split] = {"text": texts, "labels": labels.tolist(), "events": events.tolist()}
    data["meta"] = {"source": "synthetic", "revision": f"seed{seed}", "sha256": humaid_hash(data)}
    write_json(data_dir(cache_dir, "humaid", tiny=True) / "data.json", data)
    return data


def build_tiny_eurosat(cache_dir, seed=0):
    """A stand-in for EuroSAT: 8 x 8 images whose mean colour depends on the class."""
    rng = np.random.default_rng(seed + 1)
    palette = rng.integers(40, 216, size=(TINY_IMAGE_CLASSES, 1, 1, 3))
    arrays = {}
    for split in SPLITS:
        labels = rng.integers(0, TINY_IMAGE_CLASSES, size=TINY_COUNTS[split])
        images = np.clip(palette[labels] + rng.normal(0, 25, size=(len(labels), 8, 8, 3)), 0, 255).astype(np.uint8)
        arrays[split] = (images, labels.astype(np.int64))
    write_eurosat(data_dir(cache_dir, "eurosat", tiny=True), arrays,
                  {"source": "synthetic", "revision": f"seed{seed}", "classes": [f"class_{c}" for c in range(TINY_IMAGE_CLASSES)]})


def build_tiny_assets(root, seed=0):
    """Create the tiny models and datasets under ``root`` (existing ones are reused).

    Returns ``{task: fields}``: the ``SatConfig`` fields that route a run of that task to the assets.
    """
    root = Path(root)
    cache = root / "cache"
    text_model, image_model = root / "tiny-opt", root / "tiny-vit"
    if not (text_model / "config.json").exists():
        build_tiny_model(text_model, "opt", seed)
    if not (image_model / "config.json").exists():
        build_tiny_vit(image_model, seed)
    if not (data_dir(cache, "humaid", tiny=True) / "data.json").exists():
        build_tiny_humaid(cache, seed)
    if not (data_dir(cache, "eurosat", tiny=True) / "meta.json").exists():
        build_tiny_eurosat(cache, seed)
    common = {"tiny": True, "cache_dir": str(cache), "device": "cpu", "eval_batch_size": 128}
    return {"t1": {**common, "task": "t1", "model_path": str(text_model)},
            "t2": {**common, "task": "t2", "model_path": str(image_model)}}
