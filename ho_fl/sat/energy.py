"""Compute time, power caps and battery accounting (brief v2, sections 5.3 and 5.4).

Compute time uses the forward-equivalent cost model of ``ho_fl.costs`` and a nominal forward time per task, so the
simulated clock does not depend on the machine that runs the simulation. A lower power cap (eclipse) stretches the
time in proportion; a cap of zero means the satellite cannot train (forward-only satellites in eclipse).
"""
import numpy as np

from ..costs import forward_equivalents
from .config import CLASS_SPECS

MIN_CHARGE = 0.3   # a satellite trains only if that leaves at least this share of its battery


def power_cap_w(kind, sunlit):
    spec = CLASS_SPECS[kind]
    return spec.power_sun_w if sunlit else spec.power_eclipse_w


def compute_time_s(kind, sunlit, cut, depth, E, q, zo, t_fwd, slowdown):
    """Seconds of one participation: ``E * phi(cut, q) * t_fwd * slowdown * P_sun / P(t)``; ``None`` without power."""
    power = power_cap_w(kind, sunlit)
    if power <= 0:
        return None
    work = forward_equivalents("hofl", cut=cut, depth=depth, E=E, q=q, zo=zo)
    return work * t_fwd * slowdown[kind] * CLASS_SPECS[kind].power_sun_w / power


def compute_energy_j(kind, sunlit, seconds):
    return power_cap_w(kind, sunlit) * seconds


class Batteries:
    """Charge of every satellite in Wh. Backbone satellites are not accounted: their capacity is zero here and
    every request is feasible."""

    def __init__(self, roles):
        self.kinds = [roles.kind(sid) for sid in range(roles.count)]
        self.capacity = np.array([CLASS_SPECS[kind].battery_wh for kind in self.kinds])
        self.solar_w = np.array([CLASS_SPECS[kind].solar_w for kind in self.kinds])
        self.bus_w = np.array([CLASS_SPECS[kind].bus_w for kind in self.kinds])
        self.level = self.capacity.copy()

    def feasible(self, sid, energy_j):
        """Whether spending ``energy_j`` now keeps the battery at or above the minimum charge."""
        capacity = self.capacity[sid]
        return capacity == 0 or self.level[sid] - energy_j / 3600 >= MIN_CHARGE * capacity

    def share(self, sid):
        """Charge as a share of the capacity; ``None`` for a satellite whose battery is not accounted."""
        return float(self.level[sid] / self.capacity[sid]) if self.capacity[sid] else None

    def update(self, sun_seconds, round_s, spent_j):
        """End of a round: solar input over the sunlit seconds, the platform load and what the satellite spent."""
        gained = (self.solar_w * np.asarray(sun_seconds) - self.bus_w * round_s - np.asarray(spent_j)) / 3600
        self.level = np.clip(self.level + gained, 0.0, self.capacity)
