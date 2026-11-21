"""Configuration defaults and validation. The defaults of the upstream fields are frozen here (invariant I1)."""
import json
from dataclasses import asdict

import pytest

from ho_fl.config import Config, parse_config, parse_rounds

# Defaults of HKU-WILL-Lab/ho-fl@ca74bed9. They must never change.
UPSTREAM_DEFAULTS = {
    "model": "opt", "dataset": "sst2", "method": "hofl", "beta": 0.75, "seed": 42, "partition": "iid", "n": 30,
    "k": 6, "E": 5, "q": 2, "batch_size": 16, "rounds": 160, "mu": 0.001, "lr": 0.00001, "prox_lambda": 0.001,
    "weight_decay": 0.0005, "eps": 1e-8, "sampler": "dep_round_v1", "manifest": "", "model_path": "",
    "data_path": "", "token_cache": "", "cache_dir": "cache", "output": "outputs/run", "device": "cuda",
    "num_threads": 4, "eval_batch_size": 64, "evaluate_every": 0, "local_files_only": False,
}
# Options added by RHO-FFT, with the value that keeps the upstream behaviour. Register every new field here.
NEW_DEFAULTS = {"save_state_rounds": "", "init_state": "", "zo_mode": "on", "server_lr": 1.0, "agg_electorate": "all",
                "agg_weight": "cohort", "fo_quorum": 1, "thin_policy": "all", "tier_quota": "", "in_tier": "rotate",
                "rotate_window": 20, "layer_stats": False, "probe_rounds": "", "agg_stat": "mean", "agg_unit": "layer",
                "norm_clip": 0.0, "schedule_file": "", "fault_spec": "", "on_nonfinite": "raise", "per_class_eval": False}


def test_defaults_are_the_upstream_defaults_plus_registered_new_fields():
    assert asdict(Config()) == {**UPSTREAM_DEFAULTS, **NEW_DEFAULTS}


def test_resolve_fills_derived_values():
    cfg = Config().resolve()
    assert cfg.evaluate_every == 10 and cfg.beta == 0.75
    assert cfg.method_label == "HO-FL (β=0.75)"
    assert Config(E=1).resolve().evaluate_every == 50
    assert Config(sampler="auto").resolve().sampler == "dep_round_v1"


@pytest.mark.parametrize("method, label", [("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("decomfl", "DeComFL-AdamW")])
def test_beta_is_zero_for_methods_other_than_hofl(method, label):
    cfg = Config(method=method, beta=0.75).resolve()
    assert cfg.beta == 0.0 and cfg.method_label == label


@pytest.mark.parametrize("fields", [
    {"model": "gpt2"}, {"dataset": "mnli"}, {"method": "scaffold"}, {"n": 20}, {"k": 0}, {"k": 31}, {"E": 0},
    {"beta": 1.5}, {"mu": 0.0}, {"lr": 0.0}, {"partition": "dirichlet_05"}, {"sampler": "tier_quota"},
    {"sampler": "historical_torch"},  # beta > 0 requires dep_round_v1
    {"save_state_rounds": "0"}, {"save_state_rounds": "161"}, {"save_state_rounds": "ten"},
])
def test_resolve_rejects_invalid_settings(fields):
    with pytest.raises(ValueError):
        Config(**fields).resolve()


def test_resolve_accepts_non_iid_only_for_ag_news_and_uniform_historical_samplers():
    assert Config(dataset="ag_news", partition="dirichlet_05").resolve().partition == "dirichlet_05"
    assert Config(sampler="historical_torch", beta=0.0).resolve().sampler == "historical_torch"
    assert Config(save_state_rounds="40,160").resolve().save_state_rounds == "40,160"


def test_parse_rounds():
    assert parse_rounds("") == []
    assert parse_rounds("3,1,3") == [1, 3]
    assert parse_rounds(" 2 , 5 ,") == [2, 5]
    with pytest.raises(ValueError):
        parse_rounds("a")


def test_parse_config_merges_file_and_command_line(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"method": "fedavg", "rounds": 7, "seed": 43}))
    cfg = parse_config(["--config", str(path), "--rounds", "3", "--save-state-rounds", "1,3", "--local-files-only"])
    assert (cfg.method, cfg.rounds, cfg.seed, cfg.save_state_rounds, cfg.local_files_only) == ("fedavg", 3, 43, "1,3", True)
    assert parse_config(["--no-local-files-only"]).local_files_only is False


def test_parse_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"no_such_field": 1}))
    with pytest.raises(ValueError):
        parse_config(["--config", str(path)])


@pytest.mark.parametrize("fields", [
    {"zo_mode": "half"}, {"zo_mode": "off", "method": "fedavg"}, {"server_lr": 0.0}, {"fo_quorum": 0},
    {"agg_electorate": "robust"}, {"agg_weight": "mean"}, {"thin_policy": "drop"},
    {"sampler": "tier_quota", "tier_quota": "3,1"}, {"sampler": "tier_quota", "tier_quota": "3,2,2"},
    {"sampler": "tier_quota", "tier_quota": "3,1,x"}, {"sampler": "tier_quota", "tier_quota": "11,0,0", "k": 11},
    {"tier_quota": "3,1,2"},  # a quota without the sampler that uses it
    {"sampler": "tier_quota", "tier_quota": "3,1,2", "in_tier": "round_robin"}, {"rotate_window": 0},
    {"probe_rounds": "0"}, {"probe_rounds": "161"}, {"dataset": "squad", "layer_stats": True},
    {"dataset": "squad", "probe_rounds": "1"},
])
def test_resolve_rejects_invalid_new_settings(fields):
    with pytest.raises(ValueError):
        Config(**fields).resolve()


def test_new_samplers_and_labels():
    quota = Config(sampler="tier_quota", tier_quota="3,1,2").resolve()
    assert quota.beta == 0.0 and quota.method_label == "HO-FL (β=0) [tier_quota 3,1,2 rotate]"
    assert Config(sampler="tier_equalised").resolve().method_label == "HO-FL (β=0.75) [tier_equalised]"
    assert Config(sampler="tier_equalised").resolve().resolve().beta == 0.75  # resolve() is idempotent
    arm_c = Config(beta=0.0, agg_electorate="fo", agg_weight="count").resolve()
    assert arm_c.aggregation_name == "fo_mean" and arm_c.method_label == "HO-FL (β=0) [fo_mean]"
    arm_b = Config(agg_electorate="fo", zo_mode="off").resolve()
    assert arm_b.aggregation_name == "fo_diluted" and arm_b.method_label == "HO-FL (β=0.75) [fo_diluted, zo=off]"
    assert Config(server_lr=3.0).resolve().method_label == "HO-FL (β=0.75) [server_lr=3]"
    odd = Config(agg_electorate="fo", fo_quorum=3, thin_policy="hold").resolve()
    assert odd.aggregation_name == "agg=fo/cohort/quorum3/hold"
    assert Config(method="fedavg").resolve().variant_label == "" and Config().aggregation_name == "mean_all"
