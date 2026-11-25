"""The methods of the comparison as sets of switches (brief v2, section 6.4).

    m0  ideal centralised training: every invited satellite backpropagates through the whole model and a virtual
        server averages; no memory, energy or link constraint. The upper bound.
    m2  in-orbit first-order training: every participant must backpropagate through the whole model, so only the
        satellites whose memory allows that (the backbone satellites) take part; plane sums are exchanged exactly.
    m4  zeroth-order flooding: every block of every satellite is forward-only (the head of a vision task is still
        trained by backpropagation, otherwise a randomly initialised head could not be trained at all).
    m7  the full method: the cut follows the class and the memory state of the satellite and moves up when the
        first-order update would miss the deadline; both channels; protection.

All of them average with the rule of HO-FL (sum over the participants divided by their number) and differ in who
takes part, where the cut is and whether the zeroth-order part exists. m1, m3, m5 and m6 follow in milestone P-S4.

Variants change single switches of a method (``resolve``). They exist because of what the session of 2026-11-25
measured (docs/reports/P-S3.md, section 4.3): the divide-by-everyone rule shrinks the first-order step of a block
to the share of the participants that can backpropagate through it (39% in the lowest blocks of the full method),
and a zeroth-order update along random directions is a vector as long as a first-order update with next to no
direction in it (cosine 0.0005 to the first-order part).

    fo_mean     the first-order sum of a tensor is divided by the number of participants that sent a first-order
                update for it, not by everyone: a block keeps its full first-order step however many satellites
                cannot backpropagate through it. The same arithmetic as before where everyone sends one (m0, m2).
    zo_zeroed   what the zeroth-order part would add to the model is left out; who takes part, what they compute
                and what is divided by what stay as they are (the control of section 6.7 of the brief).
    line        the forward-only budget is spent on one question forward passes answer well: how far to go. A
                satellite without backpropagation (cut at the model depth) evaluates its loss on its own batches at
                the model it received and one and two more of the previous global update further, and floods the
                three numbers; every replica fits a parabola to the pooled numbers and extends the round's update
                along the previous one towards the measured minimum, by at most one previous update
                (``aggregate.step_length``, ``extension``): a momentum with a measured coefficient.
                Satellites that backpropagate through part of the model leave the blocks below their cut alone. With
                ``zo_zeroed`` the probes are still taken and logged, and the extension is left out.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    cut_rule: str       # "zero": cut 0 for everyone; "depth": cut L for everyone; "table": by class and memory state
    zo: bool            # participants run the zeroth-order part below their cut
    exchange: str       # "server": a virtual server, no network; "orbit": plane sums, laser ring, broadcast and flooding
    feasibility: bool   # energy and memory decide whether an invited satellite can take part
    c1: bool            # orbit-aware configuration: the cut moves up the ladder when the update would be late
    c3: bool            # protection measures (their checks arrive with the faults in milestone P-S4)
    fo_average: str = "all"   # what the first-order sum of a tensor is divided by: "all" counted participants, or its "senders"
    zo_weight: float = 1.0    # weight of the zeroth-order part in the update; 0 leaves it out
    zo_search: bool = False   # forward-only satellites probe the step length along the previous update (variant "line")

    @property
    def upstream_rule(self):
        """Whether a round is aggregated exactly as upstream HO-FL does: one sum, divided by the participants."""
        return self.fo_average == "all" and self.zo_weight == 1.0 and not self.zo_search


METHODS = {
    "m0": Method("m0", "ideal centralised", "zero", False, "server", False, False, False),
    "m2": Method("m2", "in-orbit first-order", "zero", False, "orbit", True, False, True),
    "m4": Method("m4", "zeroth-order flooding", "depth", True, "orbit", True, False, True),
    "m7": Method("m7", "RHO-FFT", "table", True, "orbit", True, True, True),
}

VARIANTS = {
    "fo_mean": {"fo_average": "senders"},
    "zo_zeroed": {"zo_weight": 0.0},
    "line": {"zo_search": True},
}


def resolve(key, variants=()):
    """The method ``key`` with the switches of the given variants; names that are not method variants (the data
    variant ``zonly``) are ignored. Raises ``ValueError`` for a variant that has nothing to act on."""
    method = METHODS[key]
    chosen = [name for name in variants if name in VARIANTS]
    for name in chosen:
        if name in ("zo_zeroed", "line") and not method.zo:
            raise ValueError(f"Variant {name!r} needs a method with a zeroth-order part; {key} has none")
        method = replace(method, **VARIANTS[name])
    if chosen:
        method = replace(method, label=f"{method.label} [{', '.join(sorted(chosen))}]")
    return method
