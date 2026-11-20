"""Synthetic assets for CPU-only tests and dry runs: a random tiny model, a matching tokenizer and token caches.

Nothing in this module is used by training. The assets let the whole training loop run in seconds without network
access or a GPU, which is what the unit tests and the dry run of the experiment pipelines need. Numbers produced with
these assets say nothing about model quality; they only exercise the code path.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .config import SPECS

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>"]
VERBALIZER_WORDS = ["bad", "good", "no", "yes", "A", "B", "C", "D"]
VOCAB_SIZE = 64
FIRST_WORD_ID = len(SPECIAL) + len(VERBALIZER_WORDS)
CLASSES = {"sst2": 2, "boolq": 2, "sciq": 4, "ag_news": 4}


def build_tiny_model(directory, model="opt", seed=0):
    """Save a randomly initialised tiny model and its tokenizer to ``directory`` (loadable through ``--model-path``)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

    if model != "opt":
        raise ValueError("Tiny assets are only defined for the OPT architecture")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    words = SPECIAL + VERBALIZER_WORDS + [f"w{i}" for i in range(FIRST_WORD_ID, VOCAB_SIZE)]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="</s>", eos_token="</s>", unk_token="<unk>",
                            pad_token="<pad>").save_pretrained(directory)
    config = OPTConfig(vocab_size=VOCAB_SIZE, hidden_size=32, num_hidden_layers=SPECS[model][2], ffn_dim=64,
                       num_attention_heads=4, word_embed_proj_dim=32, max_position_embeddings=64, dropout=0.0,
                       attention_dropout=0.0, activation_dropout=0.0, layerdrop=0.0, pad_token_id=1, bos_token_id=2,
                       eos_token_id=2)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        OPTForCausalLM(config).save_pretrained(directory)
    finally:
        torch.random.set_rng_state(rng_state)
    return directory


def _sequences(rng, count, low=4, high=10):
    lengths = rng.integers(low, high + 1, size=count)
    tokens = rng.integers(FIRST_WORD_ID, VOCAB_SIZE, size=(count, high))
    return [row[:n].tolist() for row, n in zip(tokens, lengths)]


def build_token_cache(path, dataset, seed=0):
    """Write a synthetic token cache with the example counts ``load_tokens`` asserts for ``dataset``."""
    from .data import DATASETS

    train_count, validation_count = DATASETS[dataset][-2:]
    rng = np.random.default_rng(seed)
    result = {}
    for split, count in (("train", train_count), ("validation", validation_count)):
        if dataset == "squad":
            answers = _sequences(rng, count, 1, 3)
            result[split] = {"input_ids": _sequences(rng, count), "answer_ids": [a + [2] for a in answers],
                             "ids": [f"{split}-{i}" for i in range(count)], "references": [["w"]] * count,
                             "reference_visible": [True] * count}
        else:
            result[split] = {"input_ids": _sequences(rng, count),
                             "labels": rng.integers(0, CLASSES[dataset], size=count).tolist()}
    if dataset == "squad":
        result["evaluation_indices"] = list(range(8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


def build_tiny_assets(root, datasets=("sst2",), model="opt", seed=0):
    """Create the tiny model and one token cache per dataset under ``root``; existing files are reused.

    Returns ``{dataset: overrides}`` where ``overrides`` are ``Config`` fields that route a run to the assets.
    """
    root = Path(root)
    model_dir = root / f"tiny-{model}"
    if not (model_dir / "config.json").exists():
        build_tiny_model(model_dir, model, seed)
    overrides = {}
    for dataset in datasets:
        cache = root / f"tiny-{dataset}-cache.json"
        if not cache.exists():
            build_token_cache(cache, dataset, seed)
        overrides[dataset] = {"model": model, "dataset": dataset, "model_path": str(model_dir),
                              "token_cache": str(cache), "cache_dir": str(root / "hf-cache"), "device": "cpu"}
    return overrides
