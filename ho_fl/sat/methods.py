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
"""
from dataclasses import dataclass


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


METHODS = {
    "m0": Method("m0", "ideal centralised", "zero", False, "server", False, False, False),
    "m2": Method("m2", "in-orbit first-order", "zero", False, "orbit", True, False, True),
    "m4": Method("m4", "zeroth-order flooding", "depth", True, "orbit", True, False, True),
    "m7": Method("m7", "RHO-FFT", "table", True, "orbit", True, True, True),
}
