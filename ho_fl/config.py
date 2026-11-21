import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SPECS = {
    "opt": ("facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6", 12, [3, 6, 9]),
    "qwen": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", 28, [8, 16, 24]),
    "smollm3": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", 36, [9, 18, 27]),
}


@dataclass
class Config:
    model: str = "opt"
    dataset: str = "sst2"
    method: str = "hofl"
    beta: float = 0.75
    seed: int = 42
    partition: str = "iid"
    n: int = 30
    k: int = 6
    E: int = 5
    q: int = 2
    batch_size: int = 16
    rounds: int = 160
    mu: float = 0.001
    lr: float = 0.00001
    prox_lambda: float = 0.001
    weight_decay: float = 0.0005
    eps: float = 1e-8
    sampler: str = "dep_round_v1"
    manifest: str = ""
    model_path: str = ""
    data_path: str = ""
    token_cache: str = ""
    cache_dir: str = "cache"
    output: str = "outputs/run"
    device: str = "cuda"
    num_threads: int = 4
    eval_batch_size: int = 64
    evaluate_every: int = 0
    local_files_only: bool = False
    save_state_rounds: str = ""
    init_state: str = ""
    zo_mode: str = "on"
    server_lr: float = 1.0
    agg_electorate: str = "all"
    agg_weight: str = "cohort"
    fo_quorum: int = 1
    thin_policy: str = "all"
    tier_quota: str = ""
    in_tier: str = "rotate"
    rotate_window: int = 20
    layer_stats: bool = False
    probe_rounds: str = ""
    agg_stat: str = "mean"
    agg_unit: str = "layer"
    norm_clip: float = 0.0
    schedule_file: str = ""
    fault_spec: str = ""
    on_nonfinite: str = "raise"
    per_class_eval: bool = False
    uniform_cut: int = 0
    eval_initial: bool = True
    attribution_every: int = 0
    dose_probe: bool = False
    probe_val_examples: int = 1024

    @property
    def betas(self):
        return (0.9, 0.999)

    amsgrad = False
    foreach = True
    fused = False

    @property
    def max_length(self):
        return {"sst2": 128, "boolq": 256, "sciq": 256, "ag_news": 128, "squad": 576}[self.dataset]

    @property
    def verbalizers(self):
        return {"sst2": [" bad", " good"], "boolq": [" no", " yes"], "sciq": [" A", " B", " C", " D"], "ag_news": [" A", " B", " C", " D"], "squad": []}[self.dataset]

    @property
    def method_label(self):
        return (f"HO-FL (\u03b2={self.beta:g})" if self.method == "hofl" else {"fedavg": "FedAvg", "fedprox": "FedProx", "decomfl": "DeComFL-AdamW"}[self.method]) + self.variant_label

    @property
    def aggregation_name(self):
        """Name of the aggregation configuration in the table of the brief, or a spelled-out description."""
        for name, fields in NAMED_AGGREGATIONS.items():
            if all(getattr(self, key) == value for key, value in fields.items()):
                return name
        return (f"agg={self.agg_electorate}/{self.agg_weight}/quorum{self.fo_quorum}/{self.thin_policy}"
                + (f"/{self.agg_stat}" if self.agg_stat != "mean" else "") + (f"/{self.agg_unit}" if self.agg_unit != "layer" else "")
                + (f"/clip{self.norm_clip:g}" if self.norm_clip else ""))

    @property
    def variant_label(self):
        """Suffix that names every setting that changes the algorithm; empty for the upstream configuration."""
        parts = []
        if self.sampler == "tier_equalised":
            parts.append("tier_equalised")
        if self.sampler == "tier_quota":
            parts.append(f"tier_quota {self.tier_quota} {self.in_tier}")
        if self.sampler == "schedule_file":
            parts.append("schedule_file")
        if self.uniform_cut:
            parts.append(f"uniform cut {self.uniform_cut}")
        if self.fault_spec:
            parts.append("faults")
        if self.aggregation_name != "mean_all":
            parts.append(self.aggregation_name)
        if self.zo_mode != "on":
            parts.append(f"zo={self.zo_mode}")
        if self.server_lr != 1.0:
            parts.append(f"server_lr={self.server_lr:g}")
        return f" [{', '.join(parts)}]" if parts else ""

    def resolve(self):
        if self.model not in SPECS or self.dataset not in {"sst2", "boolq", "sciq", "ag_news", "squad"}:
            raise ValueError("Unsupported model or dataset")
        if self.method not in {"fedavg", "fedprox", "decomfl", "hofl"}:
            raise ValueError("Unsupported method")
        if self.partition not in {"iid", "dirichlet_1", "dirichlet_05", "dirichlet_01"} or (self.dataset != "ag_news" and self.partition != "iid"):
            raise ValueError("Only AG News supports non-IID partitions")
        if self.n != 30 or not 1 <= self.k <= self.n or min(self.E, self.q, self.batch_size, self.rounds) < 1:
            raise ValueError("Three resource tiers require N=30; counts must be positive")
        if not 0 <= self.beta <= 1 or self.mu <= 0 or self.lr <= 0:
            raise ValueError("Invalid beta, smoothing radius, or learning rate")
        if self.method != "hofl":
            self.beta = 0.0
        if self.sampler == "auto":
            self.sampler = "dep_round_v1"
        if self.sampler not in {"dep_round_v1", "historical_torch", "historical_numpy_uniform", "tier_equalised", "tier_quota", "schedule_file"}:
            raise ValueError("Unsupported sampler")
        if self.sampler in {"tier_quota", "schedule_file"}:
            self.beta = 0.0
        if (self.sampler == "schedule_file") != bool(self.schedule_file):
            raise ValueError("schedule_file is required by, and only used with, sampler schedule_file")
        if self.beta and self.sampler not in {"dep_round_v1", "tier_equalised"}:
            raise ValueError("Resource-preference sampling requires DepRound")
        if self.evaluate_every == 0:
            self.evaluate_every = max(1, 50 // self.E)
        if any(not 1 <= r <= self.rounds for r in parse_rounds(self.save_state_rounds)):
            raise ValueError("save_state_rounds must lie in [1, rounds]")
        if self.zo_mode not in {"on", "off"} or (self.zo_mode == "off" and self.method != "hofl"):
            raise ValueError("zo_mode is on or off; off is only defined for method hofl")
        if self.server_lr <= 0 or self.fo_quorum < 1:
            raise ValueError("server_lr must be positive and fo_quorum at least 1")
        if self.agg_electorate not in {"all", "fo"} or self.agg_weight not in {"cohort", "count"} or self.thin_policy not in {"all", "dilute", "hold"}:
            raise ValueError("Unsupported aggregation setting")
        quota = parse_quota(self.tier_quota)
        if (self.sampler == "tier_quota") != bool(quota):
            raise ValueError("tier_quota is required by, and only used with, sampler tier_quota")
        if quota and (len(quota) != 3 or sum(quota) != self.k or not all(0 <= count <= self.n // 3 for count in quota)):
            raise ValueError("tier_quota needs three counts for T1,T2,T3 that sum to k")
        if self.in_tier not in {"rotate", "random", "fixed"} or self.rotate_window < 1:
            raise ValueError("Unsupported in_tier rule or rotate_window")
        if any(not 1 <= r <= self.rounds for r in parse_rounds(self.probe_rounds)):
            raise ValueError("probe_rounds must lie in [1, rounds]")
        if (self.layer_stats or self.probe_rounds.strip()) and self.dataset == "squad":
            raise ValueError("Probes are defined for the classification datasets only")
        if self.agg_stat not in {"mean", "geomed", "cwmed", "trimmed"} or self.agg_unit not in {"layer", "segment", "whole"} or self.norm_clip < 0:
            raise ValueError("Unsupported agg_stat, agg_unit or norm_clip")
        if self.agg_unit == "whole" and self.agg_electorate != "all":
            raise ValueError("agg_unit=whole treats the model as one vector and only goes with agg_electorate=all")
        if self.on_nonfinite not in {"raise", "stop"}:
            raise ValueError("on_nonfinite is raise or stop")
        if (self.fault_spec or self.per_class_eval) and self.dataset == "squad":
            raise ValueError("Faults and per-class evaluation are defined for the classification datasets only")
        if self.uniform_cut and (self.method != "hofl" or self.uniform_cut not in SPECS[self.model][3]):
            raise ValueError("uniform_cut gives every client the same cut; it must be one of the model's tier cuts and needs method hofl")
        if self.attribution_every < 0 or self.probe_val_examples < 1:
            raise ValueError("attribution_every must be non-negative and probe_val_examples positive")
        if self.dose_probe and (self.method != "hofl" or self.sampler != "schedule_file" or self.uniform_cut):
            raise ValueError("dose_probe needs method hofl and the cohorts of a dose schedule (sampler schedule_file)")
        if (self.attribution_every or self.dose_probe) and self.dataset == "squad":
            raise ValueError("Probes are defined for the classification datasets only")
        return self


# The aggregation configurations of the brief (section 6.4), as config fields. agg_weight only matters for the
# mean; fo_quorum and thin_policy only when the electorate is restricted to first-order reports.
_PLAIN = {"agg_unit": "layer", "norm_clip": 0.0}
NAMED_AGGREGATIONS = {
    "mean_all": {"agg_electorate": "all", "agg_stat": "mean", "agg_weight": "cohort", **_PLAIN},
    "fo_mean": {"agg_electorate": "fo", "agg_stat": "mean", "agg_weight": "count", "fo_quorum": 1, "thin_policy": "all", **_PLAIN},
    "fo_diluted": {"agg_electorate": "fo", "agg_stat": "mean", "agg_weight": "cohort", "fo_quorum": 1, "thin_policy": "all", **_PLAIN},
    "robust_all": {"agg_electorate": "all", "agg_stat": "geomed", **_PLAIN},
    "robust_whole": {"agg_electorate": "all", "agg_stat": "geomed", "agg_unit": "whole", "norm_clip": 0.0},
    "fo_robust": {"agg_electorate": "fo", "agg_stat": "geomed", "fo_quorum": 1, "thin_policy": "all", **_PLAIN},
    "quorum_robust": {"agg_electorate": "fo", "agg_stat": "geomed", "fo_quorum": 3, "thin_policy": "all", **_PLAIN},
}


def parse_quota(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_rounds(text):
    return sorted({int(part) for part in text.split(",") if part.strip()})


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(name, type=field.type, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    values = json.loads(path.read_text()) if path else {}
    unknown = set(values) - set(asdict(Config()))
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update({k: v for k, v in args.items() if v is not None})
    return Config(**values).resolve()
