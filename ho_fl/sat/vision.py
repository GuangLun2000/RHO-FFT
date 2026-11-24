"""The vision task: ViT-B/16 with LoRA and a trainable classification head (brief v2, section 7.4).

The model is split like the causal language models of ``ho_fl.split``: ``ViTPartition`` runs the blocks below a
cut (``forward_prefix``) and the blocks above it followed by the head (``forward_suffix``), so ``hybrid_step``
trains it unchanged. Two conventions make that possible:

- a batch carries the pixel tensor under the key ``input_ids`` and ``None`` as ``attention_mask``, because those are
  the two things ``hybrid_step`` hands to the partition;
- ``forward_suffix`` returns logits of shape ``[batch, 1, classes]``, so the ``[:, -1]`` that picks the last token
  of a language model picks the only position here.

The head is registered with ``ho_fl.model.register_head`` as block ``depth``: it lies above every cut and is
always trained by backpropagation, also on satellites whose transformer blocks are all forward-only.
"""
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageClassification

from ..model import register_head
from ..split import Partition
from .config import SOURCES

PIXEL_MEAN, PIXEL_STD = 0.5, 0.5   # normalisation of the ImageNet-21k ViT checkpoints


class ViTPartition(Partition):
    def __init__(self, classifier_model):
        self.lm = classifier_model
        self.body = classifier_model.vit
        self.layers = self.body.encoder.layer

    def embed(self, pixels, attention_mask):
        weight = self.body.embeddings.patch_embeddings.projection.weight
        return self.body.embeddings(pixels.to(weight.dtype)), None

    def block(self, layer, hidden, context):
        output = layer(hidden)
        return output[0] if isinstance(output, tuple) else output

    def finish(self, hidden):
        return self.body.layernorm(hidden)[:, 0]

    def forward_suffix(self, hidden, context, cut, last_only=False, token_ids=None):
        for layer in self.layers[cut:]:
            hidden = self.block(layer, hidden, context)
        # The head is float32 like every trainable tensor; the backbone may be bfloat16.
        return self.lm.classifier(self.finish(hidden).float())[:, None, :]

    def forward(self, pixels, attention_mask=None, token_ids=None):
        hidden, context = self.forward_prefix(pixels, attention_mask, 0)
        return self.forward_suffix(hidden, context, 0)


def load_vit(cfg, classes, depth, dtype=torch.bfloat16):
    """``(peft model, ViTPartition, trainable parameters)``; ``cfg`` is the upstream-style config of the run.

    The seed is set before the model is created, so the randomly initialised head and the LoRA matrices are the
    same for every method of a seed. The backbone is loaded in ``dtype`` and frozen; LoRA (rank 8 on the query
    and value projections) and the head are float32.
    """
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(False)
    identifier, revision = SOURCES["vit"]
    source = cfg.model_path or identifier
    kwargs = dict(local_files_only=cfg.local_files_only, cache_dir=cfg.cache_dir)
    if not cfg.model_path:
        kwargs["revision"] = revision
    model = AutoModelForImageClassification.from_pretrained(source, num_labels=classes, torch_dtype=dtype, **kwargs)
    assert model.config.num_hidden_layers == depth
    peft = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=["query", "value"],
                                            modules_to_save=["classifier"], init_lora_weights=True, use_rslora=False,
                                            use_dora=False, fan_in_fan_out=False))
    peft.to(cfg.device)
    params = dict(sorted((n, p) for n, p in peft.named_parameters() if p.requires_grad))
    for param in params.values():
        param.data = param.data.float()
    head = [name for name in params if "lora_" not in name]
    assert head and all("classifier" in name for name in head), head
    register_head(head, depth)
    assert all(p.dtype == torch.float32 for p in params.values())
    assert all(p.dtype == dtype for p in peft.parameters() if not p.requires_grad)
    peft.eval()
    return peft, ViTPartition(peft.get_base_model()), params


def image_batch(images, labels, indices, size, device):
    """A training or evaluation batch from uint8 images ``N x H x W x 3`` held on ``device``: resized to the
    model's input size, scaled to [0, 1] and normalised. The pixels travel under ``input_ids`` (see the module
    docstring)."""
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    pixels = images[index].permute(0, 3, 1, 2).float() / 255.0
    if pixels.shape[-1] != size or pixels.shape[-2] != size:
        pixels = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False)
    return {"input_ids": (pixels - PIXEL_MEAN) / PIXEL_STD, "attention_mask": None, "labels": labels[index]}
