"""Link rates and delays (brief v2, section 5.2).

Radio links: free-space path loss and Shannon capacity. Optical links between plane roots: a fixed rate. A hop
costs the transmission time of the payload, the propagation time and one millisecond of processing; a path is the
sum of its hops (store and forward). Within a round the rate of a link is the one of the round's first slot.
"""
import math
from dataclasses import dataclass

import numpy as np

from .scenario import EDGE_LASER

C_M_S = 299792458.0
BOLTZMANN_J_K = 1.380649e-23
PROCESSING_S = 1e-3
RF_TX_POWER_W = 5.0           # electrical power drawn while a radio transmits
LASER_TX_POWER_W = 30.0       # the same for an optical terminal


@dataclass(frozen=True)
class RadioLink:
    frequency_hz: float
    bandwidth_hz: float
    tx_power_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    noise_temperature_k: float = 500.0
    other_losses_db: float = 3.0

    def snr(self, distance_km):
        path_loss = (4 * math.pi * np.asarray(distance_km, dtype=np.float64) * 1e3 * self.frequency_hz / C_M_S) ** 2
        power_w = 10 ** ((self.tx_power_dbm - 30) / 10)
        gain = 10 ** ((self.tx_gain_dbi + self.rx_gain_dbi - self.other_losses_db) / 10)
        return power_w * gain / (BOLTZMANN_J_K * self.noise_temperature_k * self.bandwidth_hz * path_loss)

    def rate_bps(self, distance_km):
        """Shannon capacity at this distance, in bit/s (a float for a scalar distance, else an array)."""
        rate = self.bandwidth_hz * np.log2(1 + self.snr(distance_km))
        return float(rate) if np.ndim(rate) == 0 else rate


def inter_satellite_radio(gain_dbi=20.0):
    """The radio links between satellites; the antenna gain is the knob of the radio sensitivity study."""
    return RadioLink(2.4e9, 20e6, 37.0, gain_dbi, gain_dbi)


GROUND_LINK = RadioLink(11e9, 250e6, 50.0, 35.0, 35.0)   # satellite-ground, used by the ground-server baseline only


def hop_delay_s(payload_bytes, rate_bps, distance_km):
    """Seconds to move ``payload_bytes`` over one link: transmission, propagation and processing."""
    return 8 * payload_bytes / rate_bps + distance_km * 1e3 / C_M_S + PROCESSING_S


class LinkBudget:
    """Rates of all edges of a trace in one round, from the distances in the round's first slot."""

    def __init__(self, trace, view, rf_gain_dbi=20.0, laser_gbps=10.0):
        self.trace = trace
        self.distance_km = view.edge_dist_km
        self.laser = trace.edge_kind == EDGE_LASER
        self.rate_bps = np.where(self.laser, laser_gbps * 1e9, inter_satellite_radio(rf_gain_dbi).rate_bps(self.distance_km))
        self.tx_power_w = np.where(self.laser, LASER_TX_POWER_W, RF_TX_POWER_W)
        self.energy_per_byte = self.tx_power_w * 8 / self.rate_bps        # joules the sender spends per byte, per edge
        ends = trace.edges.astype(int)
        self.node_energy_per_byte = (np.bincount(ends[:, 0], self.energy_per_byte, trace.count)
                                     + np.bincount(ends[:, 1], self.energy_per_byte, trace.count))   # over all links of a satellite

    def hop(self, a, b, payload_bytes):
        """Delay of one hop between neighbouring satellites ``a`` and ``b``."""
        edge = self.trace.edge(a, b)
        return hop_delay_s(payload_bytes, float(self.rate_bps[edge]), float(self.distance_km[edge]))

    def path(self, satellites, payload_bytes):
        """Delay along a path given as the satellites it visits."""
        return sum(self.hop(a, b, payload_bytes) for a, b in zip(satellites, satellites[1:]))

    def tx_energy_j(self, a, b, payload_bytes):
        """Energy the sender spends on one hop: transmit power times transmission time."""
        edge = self.trace.edge(a, b)
        return float(self.tx_power_w[edge]) * 8 * payload_bytes / float(self.rate_bps[edge])

    def worst_hop(self, edges, payload_bytes):
        """Largest hop delay over a set of edge indices (the laser ring uses its slowest edge)."""
        return max(hop_delay_s(payload_bytes, float(self.rate_bps[edge]), float(self.distance_km[edge])) for edge in edges)
