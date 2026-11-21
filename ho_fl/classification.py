import torch
import torch.nn.functional as F
from .batches import collate

@torch.no_grad()
def evaluate(wrapper, data, tokens, pad_id, batch_size, device, per_class=False):
    loss_sum, correct, count = (0.0, 0, len(data['labels']))
    hits, totals = {}, {}
    for start in range(0, count, batch_size):
        batch = collate(data, list(range(start, min(start + batch_size, count))), pad_id, device)
        logits = wrapper.forward(batch['input_ids'], batch['attention_mask'], token_ids=tokens)[:, -1].float()
        assert torch.isfinite(logits).all()
        loss_sum += F.cross_entropy(logits, batch['labels'], reduction='sum').item()
        correct += (logits.argmax(-1) == batch['labels']).sum().item()
        if per_class:
            for label, hit in zip(batch['labels'].tolist(), (logits.argmax(-1) == batch['labels']).tolist()):
                totals[label] = totals.get(label, 0) + 1
                hits[label] = hits.get(label, 0) + int(hit)
    result = {'accuracy': correct / count, 'validation_loss': loss_sum / count, 'validation_count': count}
    if per_class:
        result['per_class_accuracy'] = [hits.get(label, 0) / totals[label] if totals.get(label) else None for label in range(logits.shape[-1])]
    return result
