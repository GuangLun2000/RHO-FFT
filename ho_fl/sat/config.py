"""Configuration of the satellite scenario (brief v2, section 8.1 and appendix B).

``SatConfig`` describes one run. The functions reused from ``ho_fl`` (optimizer, hybrid step, replay, model
loading) expect the upstream ``Config``; ``SatConfig.train_config`` builds the one they see.
"""
import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..config import Config

CLASSES = ("B", "H", "Z")   # backbone, hybrid (GPU) and forward-only (inference chip) satellites; tier 0, 1, 2

# Constellations as Walker-delta inclination: N / P / F at one altitude.
SCENARIOS = {
    "s1": {"planes": 12, "per_plane": 10, "phasing": 1, "inclination_deg": 53.0, "altitude_km": 550.0},
    "s2": {"planes": 72, "per_plane": 22, "phasing": 1, "inclination_deg": 53.0, "altitude_km": 550.0},
}

# Ground stations as (latitude, longitude) in degrees. The first one is the default single station.
GROUND_STATIONS = {
    "cambridge": (52.2, 0.1),
    "punta_arenas": (-53.2, -70.9),
    "hartebeesthoek": (-25.9, 27.7),
}

# Pinned sources: (Hugging Face identifier, revision). HumAID comes from the per-event release because the
# pooled release (QCRI/HumAID-all) has no event field; the 19 event subsets add up to the same three splits.
SOURCES = {
    "humaid": ("QCRI/HumAID-events", "2e7ea23332006f075068b90d401b178d847b447a"),
    "eurosat": ("timm/eurosat-rgb", "b4e28552cd5f3932b6abc37eb20d3e84901ad728"),
    "vit": ("google/vit-base-patch16-224-in21k", "b4569560a39a0f1af58e3ddaf17facf20ab919b0"),
}


@dataclass(frozen=True)
class Task:
    key: str
    kind: str          # "text": causal LM with verbalizers; "vision": ViT with a trainable classification head
    model: str         # model key: enters the direction seeds and, for text, selects config.SPECS[model]
    dataset: str
    depth: int         # number of transformer blocks L
    ladder: tuple      # cuts of a hybrid satellite, lowest first: idle, imaging, one more step, L
    lr: float          # client learning rate, fixed per task
    t_fwd: float       # nominal seconds of one forward pass of a batch through all blocks (not a measurement)
    max_length: int = 0

    @property
    def cut_idle(self):
        return self.ladder[0]

    @property
    def cut_imaging(self):
        return self.ladder[1]


TASKS = {
    "t1": Task("t1", "text", "qwen", "humaid", 28, (8, 16, 24, 28), 1e-5, 0.04, max_length=192),
    "t2": Task("t2", "vision", "vit", "eurosat", 12, (3, 6, 9, 12), 1e-4, 0.01),
}
# The same tasks on the synthetic assets of ``ho_fl.sat.testing``: 12-block models, so the cut table is T2's.
TINY_TASKS = {
    "t1": Task("t1", "text", "opt", "humaid", 12, (3, 6, 9, 12), 1e-5, 0.04, max_length=32),
    "t2": Task("t2", "vision", "vit", "eurosat", 12, (3, 6, 9, 12), 1e-4, 0.01),
}


@dataclass(frozen=True)
class ClassSpec:
    slowdown: float          # compute time relative to a backbone satellite
    power_sun_w: float       # compute power cap in sunlight
    power_eclipse_w: float   # compute power cap in eclipse
    battery_wh: float        # 0: the battery is not accounted (backbone satellites)
    solar_w: float           # solar input in sunlight
    bus_w: float             # platform load, always on


CLASS_SPECS = {
    "B": ClassSpec(1.0, 1000.0, 300.0, 0.0, 0.0, 0.0),
    "H": ClassSpec(30.0, 40.0, 10.0, 100.0, 80.0, 30.0),
    "Z": ClassSpec(150.0, 10.0, 0.0, 40.0, 25.0, 10.0),
}

METHODS = ("m0", "m2", "m4", "m7")   # the methods implemented so far; m1, m3, m5, m6 follow in milestone P-S4
VARIANTS = ("zonly",)                # data variant; the remaining variants of section 6.7 follow with E2


@dataclass
class TrainConfig(Config):
    """The upstream ``Config`` handed to reused functions. The verbalizers and the prompt length come from the
    satellite task instead of the upstream dataset table. It is built ready to use and never resolved: the
    upstream checks only know the upstream models, datasets and device count."""
    task_verbalizers: tuple = ()
    task_max_length: int = 0

    @property
    def verbalizers(self):
        return list(self.task_verbalizers)

    @property
    def max_length(self):
        return self.task_max_length

    def resolve(self):
        raise RuntimeError("TrainConfig comes from SatConfig.train_config and is not resolved")


def _default_pi():
    return {"B": 0.5, "H": 0.125, "Z": 0.1}


def _default_slowdown():
    return {name: spec.slowdown for name, spec in CLASS_SPECS.items()}


@dataclass
class SatConfig:
    task: str = "t2"
    method: str = "m7"
    variant: str = ""                 # comma-separated variants of section 6.7
    scenario: str = "s1"
    seed: int = 42
    hours: float = 24.0               # simulated time
    round_s: int = 600
    slot_s: int = 30
    deadline_s: float = 480.0         # upper bound of the collection deadline D_up
    pi: dict = field(default_factory=_default_pi)            # invitation probability per class
    E: int = 5
    q: int = 2
    batch_size: int = 16
    mu: float = 1e-2
    lr: float = 0.0                   # client learning rate; 0 takes the task's value
    server_lr: float = 1.0
    t_fwd: float = 0.0                # nominal forward time in seconds; 0 takes the task's value
    slowdown: dict = field(default_factory=_default_slowdown)
    rf_gain_dbi: float = 20.0
    laser_gbps: float = 10.0
    gs_sites: list = field(default_factory=lambda: ["cambridge"])
    eval_every: int = 6               # rounds between evaluations: one simulated hour
    faults: str = ""                  # JSON file with the fault specification of E3
    protect: bool = True
    output: str = "outputs/sat/run"
    # Where things come from and where they run.
    trace: str = ""                   # trace directory; empty: <cache_dir>/sat/traces/<scenario>
    manifest: str = ""                # partition file; empty: <cache_dir>/sat/manifests/<task>-<variant>-s<seed>.json
    eval_split: str = "test"          # "validation" for step-size selection
    cache_dir: str = "cache"
    model_path: str = ""
    device: str = "cuda"
    num_threads: int = 4
    eval_batch_size: int = 64
    local_files_only: bool = False
    tiny: bool = False                # synthetic assets of ho_fl.sat.testing: checks the code path, not the method
    # Training constants shared with upstream HO-FL.
    weight_decay: float = 0.0005
    eps: float = 1e-8

    @property
    def spec(self):
        return (TINY_TASKS if self.tiny else TASKS)[self.task]

    @property
    def constellation(self):
        return SCENARIOS[self.scenario]

    @property
    def satellites(self):
        return self.constellation["planes"] * self.constellation["per_plane"]

    @property
    def rounds(self):
        return int(round(self.hours * 3600 / self.round_s))

    @property
    def slots_per_round(self):
        return self.round_s // self.slot_s

    @property
    def variants(self):
        return [part.strip() for part in self.variant.split(",") if part.strip()]

    @property
    def data_variant(self):
        return "zonly" if "zonly" in self.variants else "base"

    @property
    def trace_dir(self):
        return Path(self.trace) if self.trace else Path(self.cache_dir) / "sat" / "traces" / self.scenario

    @property
    def manifest_path(self):
        if self.manifest:
            return Path(self.manifest)
        tag = "-tiny" if self.tiny else ""
        return Path(self.cache_dir) / "sat" / "manifests" / f"{self.task}{tag}-{self.data_variant}-s{self.seed}.json"

    def resolve(self):
        if self.task not in TASKS or self.scenario not in SCENARIOS:
            raise ValueError("Unsupported task or scenario")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method {self.method!r}; implemented: {', '.join(METHODS)}")
        unknown = [name for name in self.variants if name not in VARIANTS]
        if unknown:
            raise ValueError(f"Unsupported variant {unknown}; implemented: {', '.join(VARIANTS)}")
        if self.scenario != "s1":
            raise ValueError("Training is defined on scenario s1 only; s2 is used for the topology computation")
        if self.lr == 0:
            self.lr = self.spec.lr
        if self.t_fwd == 0:
            self.t_fwd = self.spec.t_fwd
        if min(self.E, self.q, self.batch_size, self.eval_every, self.eval_batch_size) < 1:
            raise ValueError("Counts must be positive")
        if min(self.mu, self.lr, self.server_lr, self.t_fwd, self.hours, self.laser_gbps) <= 0:
            raise ValueError("mu, lr, server_lr, t_fwd, hours and laser_gbps must be positive")
        if self.round_s % self.slot_s or self.round_s <= 0 or not 0 < self.deadline_s <= self.round_s:
            raise ValueError("A round is a whole number of slots and the deadline lies within the round")
        if abs(self.hours * 3600 / self.round_s - self.rounds) > 1e-9 or self.rounds < 1:
            raise ValueError("hours must be a whole number of rounds")
        if set(self.pi) != set(CLASSES) or not all(0 <= value <= 1 for value in self.pi.values()):
            raise ValueError("pi needs a probability for each of B, H, Z")
        if set(self.slowdown) != set(CLASSES) or not all(value > 0 for value in self.slowdown.values()):
            raise ValueError("slowdown needs a positive factor for each of B, H, Z")
        if not self.gs_sites or any(site not in GROUND_STATIONS for site in self.gs_sites):
            raise ValueError(f"gs_sites are among {sorted(GROUND_STATIONS)}")
        if self.eval_split not in {"test", "validation"}:
            raise ValueError("eval_split is test or validation")
        if self.faults:
            raise ValueError("Fault injection belongs to milestone P-S4 and is not implemented yet")
        return self

    def train_config(self, k=1, verbalizers=()):
        """The upstream ``Config`` for the reused functions; ``k`` is the number of participants of the round."""
        spec = self.spec
        return TrainConfig(model=spec.model, dataset=spec.dataset, method="hofl", beta=0.0, seed=self.seed,
                           n=self.satellites, k=k, E=self.E, q=self.q, batch_size=self.batch_size, rounds=self.rounds,
                           mu=self.mu, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps,
                           model_path=self.model_path, cache_dir=self.cache_dir, output=self.output, device=self.device,
                           num_threads=self.num_threads, eval_batch_size=self.eval_batch_size,
                           evaluate_every=self.eval_every, local_files_only=self.local_files_only,
                           server_lr=self.server_lr, scenario=self.scenario,
                           task_verbalizers=tuple(verbalizers), task_max_length=spec.max_length)

    def to_dict(self):
        return asdict(self)


def parse_sat_config(argv=None, **overrides):
    """``SatConfig`` from command-line options named like its fields (``--server-lr 3``, ``--no-protect``), on top
    of an optional ``--config`` JSON file. Dictionaries and lists are given as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    for item in fields(SatConfig):
        name = "--" + item.name.replace("_", "-")
        if item.type is bool:
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        elif item.type in (dict, list):
            parser.add_argument(name, type=json.loads, default=None)
        else:
            parser.add_argument(name, type=item.type, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    values = json.loads(path.read_text()) if path else {}
    unknown = set(values) - {item.name for item in fields(SatConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update({key: value for key, value in args.items() if value is not None})
    values.update(overrides)
    return SatConfig(**values).resolve()
