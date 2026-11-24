"""Who is who in the constellation: satellite numbers, classes, plane roots, in-plane hops and the cut table
(brief v2, sections 1.2 and 5.3).

A satellite is numbered ``sid = plane * S + slot``. In every plane slot 0 is the backbone satellite (B) and the
aggregation root, the other even slots are hybrid satellites (H) and the odd slots are forward-only satellites (Z).
"""
from .config import CLASSES


class Roles:
    def __init__(self, planes, per_plane):
        self.planes, self.per_plane = planes, per_plane
        self.count = planes * per_plane

    def sid(self, plane, slot):
        return plane * self.per_plane + slot

    def plane(self, sid):
        return sid // self.per_plane

    def slot(self, sid):
        return sid % self.per_plane

    def kind(self, sid):
        """``"B"``, ``"H"`` or ``"Z"``."""
        slot = self.slot(sid)
        return "B" if slot == 0 else ("H" if slot % 2 == 0 else "Z")

    def tier(self, sid):
        """0, 1, 2 for B, H, Z: the value stored in ``ClientReport.tier``."""
        return CLASSES.index(self.kind(sid))

    def members(self, kind=None, plane=None):
        """Satellite numbers in ascending order, optionally of one class and one plane."""
        return [sid for sid in range(self.count)
                if (kind is None or self.kind(sid) == kind) and (plane is None or self.plane(sid) == plane)]

    def root(self, plane, root_slot=0):
        """The satellite that collects the first-order updates of ``plane``."""
        return self.sid(plane, root_slot)

    def hops_to_root(self, sid, root_slot=0):
        """In-plane hops from ``sid`` to the root of its plane along the shorter side of the ring."""
        offset = (self.slot(sid) - root_slot) % self.per_plane
        return min(offset, self.per_plane - offset)

    def path_to_root(self, sid, root_slot=0):
        """Satellites visited from ``sid`` to the root of its plane, both ends included. A satellite exactly
        opposite the root goes through the lower slots."""
        plane, offset = self.plane(sid), (self.slot(sid) - root_slot) % self.per_plane
        step = -1 if offset <= self.per_plane - offset else 1
        path = [sid]
        while offset % self.per_plane:
            offset += step
            path.append(self.sid(plane, (root_slot + offset) % self.per_plane))
        return path

    @property
    def max_hops(self):
        """Hops from the root to the farthest satellite of its plane: the length of the broadcast."""
        return self.per_plane // 2

    def grid_neighbours(self, sid):
        """The four +Grid neighbours: the two in-plane neighbours and the same slot in the two adjacent planes."""
        plane, slot = self.plane(sid), self.slot(sid)
        return [self.sid(plane, (slot + 1) % self.per_plane), self.sid(plane, (slot - 1) % self.per_plane),
                self.sid((plane + 1) % self.planes, slot), self.sid((plane - 1) % self.planes, slot)]


def memory_floor(task, kind, imaging):
    """Lowest cut the memory of a satellite allows right now (brief section 5.3): a backbone satellite
    backpropagates through everything, a hybrid one through fewer blocks while it is imaging, a forward-only one
    through none (the classification head of a vision task sits above every cut)."""
    if kind == "B":
        return 0
    if kind == "H":
        return task.cut_imaging if imaging else task.cut_idle
    return task.depth


def next_cut(task, cut):
    """The next step up the cut ladder, or ``None`` at the top (``cut == depth``)."""
    higher = [step for step in task.ladder if step > cut]
    return higher[0] if higher else None
