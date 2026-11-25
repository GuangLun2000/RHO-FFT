"""C1: what every satellite decides at the start of a round (brief v2, sections 5.6 and 6.1).

Whether a satellite is invited follows from a shared hash of the seed, the round and the satellite, so every
method sees the same invitations. An invited satellite takes part if the method lets it: the cut the method asks
for must not lie below what its memory allows right now, it must have power (a forward-only satellite has none in
eclipse) and the training must leave its battery above the minimum charge. With orbit-aware configuration the cut
moves up the ladder while the first-order update would miss the collection deadline.

Everything here is local to a satellite: its class, its own state in the first slot of the round, and link
parameters every satellite knows. Nothing is drawn from a random generator.
"""
from dataclasses import dataclass

from ..randomness import stable_seed
from .energy import compute_energy_j, compute_time_s
from .roles import memory_floor, next_cut

SKIP_REASONS = ("memory", "eclipse", "energy", "unreachable")


def lottery(seed, r, sid):
    """The shared draw U(sid, r) in [0, 1): the same for every method of a seed."""
    return stable_seed(seed, "sat-lottery", r, sid) / 2 ** 32


@dataclass
class Decision:
    sid: int
    kind: str
    lottery: float
    invited: bool
    sunlit: bool
    imaging: bool                 # hybrid satellite in sunlight over land: part of its memory is taken
    participate: bool = False
    reason: str = None            # why an invited satellite stays out: one of SKIP_REASONS
    cut: int = None
    zo: bool = False              # runs a zeroth-order part and floods its report (false for cut 0, for methods
                                  # without one and, with the step-length search, for every cut below the depth)
    tau_s: float = None           # compute time of the participation
    hops: int = 0                 # in-plane hops to the plane root
    fo_bytes: int = 0             # size of the first-order update
    arrival_s: float = None       # when the first-order update reaches the plane root; None if there is none to send
    on_time: bool = None          # whether that is before the deadline
    moved_up: int = 0             # steps the cut moved up the ladder to make the deadline
    energy_cmp_j: float = 0.0
    energy_tx_est_j: float = 0.0  # what the satellite expects to spend sending its own update and report


def decide(r, view, budget, timeline, batteries, method, cfg, backend, roles):
    """``{sid: Decision}`` for every satellite, from the state in the first slot of round ``r``.

    ``cfg`` is the ``SatConfig`` of the run, ``timeline`` the deadline of the round (``network.timeline``).
    Reasons are tried in the order memory, eclipse, energy; the first that applies is recorded.
    """
    task, decisions = backend.task, {}
    orbit = method.exchange == "orbit"
    for sid in range(roles.count):
        kind = roles.kind(sid)
        draw = lottery(cfg.seed, r, sid)
        sunlit = bool(view.sunlit[sid])
        decision = Decision(sid=sid, kind=kind, lottery=draw, invited=draw < cfg.pi[kind], sunlit=sunlit,
                            imaging=kind == "H" and sunlit and bool(view.land[sid]))
        decisions[sid] = decision
        if not decision.invited:
            continue
        floor = memory_floor(task, kind, decision.imaging)
        cut = {"zero": 0, "depth": task.depth, "table": floor}[method.cut_rule]
        if method.feasibility and cut < floor:
            decision.reason = "memory"
            continue

        def runs_zo(cut):
            # With the step-length search only a satellite without any backpropagation through the blocks spends
            # forward passes on a zeroth-order part: its q probes per batch cost what q directions did.
            return method.zo and cut > 0 and (not method.zo_search or cut >= task.depth)

        def seconds(cut, lit=sunlit):
            return compute_time_s(kind, lit, cut, task.depth, cfg.E, cfg.q, runs_zo(cut), cfg.t_fwd, cfg.slowdown)

        powered = seconds(cut) is not None
        if not powered and method.feasibility:
            decision.reason = "eclipse"
            continue
        lit = sunlit if powered else True          # a method without constraints trains as if in sunlight
        path = roles.path_to_root(sid)

        def arrival(cut):
            return seconds(cut, lit) + budget.path(path, backend.fo_bytes(cut))

        if orbit and method.c1:
            while backend.fo_bytes(cut) and arrival(cut) > timeline.deadline_s and next_cut(task, cut) is not None:
                cut = next_cut(task, cut)
                decision.moved_up += 1
        tau = seconds(cut, lit)
        fo_bytes = backend.fo_bytes(cut)
        zo = runs_zo(cut)
        energy = compute_energy_j(kind, lit, tau)
        estimate = 0.0
        if orbit:
            report = backend.zo_bytes() if zo else backend.beacon_bytes
            estimate = float(budget.node_energy_per_byte[sid]) * report
            if fo_bytes and len(path) > 1:
                estimate += budget.tx_energy_j(path[0], path[1], fo_bytes)
        if method.feasibility and not batteries.feasible(sid, energy + estimate):
            decision.reason = "energy"
            continue
        decision.participate, decision.cut, decision.zo, decision.tau_s = True, cut, zo, tau
        decision.hops, decision.fo_bytes = len(path) - 1, fo_bytes
        decision.energy_cmp_j, decision.energy_tx_est_j = energy, estimate
        if orbit and fo_bytes:
            decision.arrival_s = arrival(cut)
            decision.on_time = decision.arrival_s <= timeline.deadline_s
        elif fo_bytes:
            decision.on_time = True                # a virtual server receives everything
    return decisions
