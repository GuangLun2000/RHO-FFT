"""Client reports and helpers that group trainable tensors by layer and by segment.

A report is one client's update of the whole model for one round. Its component on layer ``l`` is first-order when
``l >= cut`` (the layer was trained by backpropagation) and zeroth-order when ``l < cut``. The server knows the cut
of every report, so it knows the order of every component.
"""
from dataclasses import dataclass

import torch

from .model import block_index


@dataclass
class ClientReport:
    cid: int
    round: int
    cut: int                    # cut used for this participation
    tier: int                   # 0, 1, 2 for T1, T2, T3: position of the client's resource cut among the model's cuts
    delta: dict                 # tensor name -> local parameters minus the global state the client started from
    scalars: list               # per local step, the zeroth-order scalars returned by the step
    batches: list               # per local step, the training example indices of the batch
    losses: list                # per local step, the detached task loss
    direction_master_seed: int
    fault: str = None
    zo: bool = True             # whether the zeroth-order update was executed (zo_mode); if not, layers below the cut did not move

    @property
    def train_loss(self):
        """Mean task loss over the local steps of this participation."""
        return torch.stack(self.losses).mean().item()


def is_first_order(cut, layer):
    """Whether a report with this cut carries a first-order component on ``layer``."""
    return layer >= cut


def layer_groups(names):
    """``{layer: [tensor names]}``, layers and names in order of first appearance."""
    groups = {}
    for name in names:
        groups.setdefault(block_index(name), []).append(name)
    return groups


def segment_bounds(cuts, depth):
    """Half-open layer ranges delimited by the distinct cuts, e.g. ``[(0, 8), (8, 16), (16, 24), (24, 28)]``."""
    edges = [0, *sorted(set(cuts)), depth]
    return [(low, high) for low, high in zip(edges, edges[1:]) if low < high]


def segment_index(layer, bounds):
    for index, (low, high) in enumerate(bounds):
        if low <= layer < high:
            return index
    raise ValueError(f"Layer {layer} is outside {bounds}")


def segment_groups(names, cuts, depth):
    """``{segment index: [tensor names]}`` for the segments of ``segment_bounds(cuts, depth)``."""
    bounds = segment_bounds(cuts, depth)
    groups = {index: [] for index in range(len(bounds))}
    for name in names:
        groups[segment_index(block_index(name), bounds)].append(name)
    return groups
