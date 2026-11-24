"""Model, data and partition of a satellite run, the local step of a participation and the evaluation
(brief v2, sections 3 and 7).

``SatBackend`` plays the role of ``ho_fl.train.Backend`` for the satellite scenario: it holds one copy of the
model, into which the state of whichever replica is training or being evaluated is restored. It differs from the
upstream backend in three ways: the prefixes exist for every cut of the task (0 and the model depth included),
the data come from the satellite partition, and the vision task has its own model, batches and head.
"""
import torch
import torch.nn.functional as F

from ..algorithms import hybrid_step, optimizer_for
from ..batches import ClientStream, collate
from ..io import tensor_hash
from ..model import load_model, partition_parameters
from .datasets import SPLITS, T1_EVAL_SUBSET, fixed_subset, load_humaid_tokens, load_manifest, load_task_data, verbalizers
from .vision import image_batch, load_vit


def classification_metrics(confusion, loss_sum):
    """Accuracy, mean loss, per-class recall and macro-F1 from a confusion matrix (rows: true class, columns:
    predicted class). Macro-F1 averages over the classes that occur in the evaluated examples."""
    confusion = confusion.double()
    support, predicted, hits = confusion.sum(1), confusion.sum(0), confusion.diagonal()
    count = float(support.sum())
    f1 = torch.where(support + predicted > 0, 2 * hits / (support + predicted).clamp_min(1), torch.zeros_like(hits))
    present = support > 0
    return {"accuracy": float(hits.sum()) / count, "loss": loss_sum / count, "count": int(count),
            "macro_f1": float(f1[present].mean()),
            "per_class_accuracy": [float(hits[c] / support[c]) if support[c] > 0 else None for c in range(len(support))]}


class SatBackend:
    def __init__(self, sat):
        """``sat`` is a resolved ``SatConfig``. Loads the model, the data and the partition of the run."""
        task = sat.spec
        raw = load_task_data(sat)
        classes = len(raw["classes"])
        manifest, manifest_id, manifest_bytes = load_manifest(sat, raw)
        if task.kind == "text":
            cfg = sat.train_config(verbalizers=verbalizers(classes))
            model, wrapper, params, tokenizer, tokens = load_model(cfg)
            assert len(set(tokens.tolist())) == classes, "verbalizers must be distinct single tokens"
            cached = load_humaid_tokens(sat.cache_dir, raw, tokenizer, task.model, task.max_length, sat.tiny)
            data = {split: cached[split] for split in SPLITS}
        else:
            cfg = sat.train_config()
            model, wrapper, params = load_vit(cfg, classes, task.depth)
            tokenizer, tokens = None, None
            data = {split: {"images": torch.as_tensor(raw[split]["images"]).to(cfg.device),
                            "labels": torch.as_tensor(raw[split]["labels"], dtype=torch.long).to(cfg.device)} for split in SPLITS}
        self._assemble(task, cfg, model, wrapper, params, tokenizer, tokens, data, manifest["satellites"], classes)
        self.sat, self.manifest, self.manifest_id, self.manifest_bytes = sat, manifest, manifest_id, manifest_bytes
        self.data_sha256, self.class_names = raw["meta"]["sha256"], list(raw["classes"])
        # Evaluation sets (brief 7.1, 7.2): HumAID curves use a fixed subset of 4,000 examples, chosen by name
        # only, and the last evaluation of a test run uses the whole split; EuroSAT always uses the whole split.
        self.eval_split = sat.eval_split
        count = self.split_size(self.eval_split)
        subset = task.dataset == "humaid" and count > T1_EVAL_SUBSET
        self.curve_indices = fixed_subset(count, T1_EVAL_SUBSET, f"t1-{self.eval_split}-subset") if subset else None
        self.final_indices = self.curve_indices if self.eval_split == "validation" else None
        held = manifest["held_out"]
        self.held_out_indices = held[f"{self.eval_split}_indices"] if held else None

    @classmethod
    def from_parts(cls, task, cfg, model, wrapper, params, tokenizer, tokens, data, records, classes):
        """A backend around an already loaded model and data (the equivalence tests build one around the parts of
        an upstream ``Backend``). ``records`` are the partition records: ``indices`` and ``data_seed`` each."""
        backend = cls.__new__(cls)
        backend._assemble(task, cfg, model, wrapper, params, tokenizer, tokens, data, records, classes)
        return backend

    def _assemble(self, task, cfg, model, wrapper, params, tokenizer, tokens, data, records, classes):
        self.task, self.cfg, self.model, self.wrapper, self.params = task, cfg, model, wrapper, params
        self.tokenizer, self.tokens, self.data, self.classes = tokenizer, tokens, data, classes
        self.initial_hash = tensor_hash(params)
        self.streams = [ClientStream(record, cfg.batch_size) for record in records]
        self.kinds = [record.get("kind") for record in records]
        self.cuts = sorted({0, *task.ladder})
        self.bottoms = {cut: partition_parameters(params, cut)[0] for cut in self.cuts}
        self.dimensions = {cut: sum(p.numel() for p in bottom.values()) for cut, bottom in self.bottoms.items()}
        self.total = sum(p.numel() for p in params.values())
        self.image_size = wrapper.lm.config.image_size if task.kind == "vision" else None
        self.processed_tokens = 0

    # --- sizes -----------------------------------------------------------------------------------------------------

    def fo_bytes(self, cut):
        """Bytes of the first-order update of a participation with this cut: float32 tensors of the blocks at or
        above the cut, and the head if the task has one."""
        return 4 * (self.total - self.dimensions[cut])

    @property
    def full_bytes(self):
        return 4 * self.total

    def zo_bytes(self):
        """Bytes of a zeroth-order report: an 18-byte header and one float32 scalar per local step and direction."""
        return 18 + 4 * self.cfg.E * self.cfg.q

    def split_size(self, split):
        return len(self.data[split]["labels"])

    # --- training --------------------------------------------------------------------------------------------------

    def optimizer(self):
        return optimizer_for(self.params, self.cfg)

    def batch(self, split, indices):
        if self.task.kind == "text":
            return collate(self.data[split], indices, self.tokenizer.pad_token_id, self.cfg.device)
        return image_batch(self.data[split]["images"], self.data[split]["labels"], indices, self.image_size, self.cfg.device)

    def local_step(self, sid, indices, optimizer, directions, e, *, cut, zo=True):
        """One local step on the training examples ``indices``, with the blocks below ``cut`` forward-only."""
        batch = self.batch("train", indices)
        if self.task.kind == "text":
            self.processed_tokens += int(batch["attention_mask"].sum().item())
        return hybrid_step(self.cfg, self.wrapper, self.params, self.bottoms[cut], batch, self.tokens, optimizer, directions, e, zo=zo, cut=cut)

    # --- evaluation ------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, split, indices=None):
        """Metrics of the parameters currently in the model on ``split`` (all of it, or the given examples).
        Raises ``AssertionError`` on non-finite logits, like the upstream evaluation."""
        indices = list(range(self.split_size(split))) if indices is None else list(indices)
        confusion = torch.zeros(self.classes, self.classes, dtype=torch.long, device=self.cfg.device)
        loss_sum, size = 0.0, self.cfg.eval_batch_size
        for start in range(0, len(indices), size):
            batch = self.batch(split, indices[start:start + size])
            logits = self.wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=self.tokens)[:, -1].float()
            assert torch.isfinite(logits).all()
            loss_sum += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
            confusion.index_put_((batch["labels"], logits.argmax(-1)), torch.ones_like(batch["labels"]), accumulate=True)
        return classification_metrics(confusion.cpu(), loss_sum)
