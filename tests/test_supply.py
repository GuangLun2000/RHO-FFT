"""Supply metrics against appendix A of the brief (analytic rows exactly, simulated rows within its tolerance)."""
import json

import numpy as np
import pytest

from ho_fl import supply
from ho_fl.config import SPECS


@pytest.fixture(scope="module")
def qwen():
    return supply.plan("qwen", seeds=60, rounds=160)


@pytest.mark.parametrize("name", sorted(supply.APPENDIX_A))
def test_analytic_rows_of_appendix_a(qwen, name):
    reference, result = supply.APPENDIX_A[name], qwen[name]["analytic"]
    s1, s2, s3 = result["segments"][1:]
    assert result["segments"][0]["fo_capable"] == 0 and result["segments"][0]["width_mean"] == 0.0
    assert [s1["fo_capable"], s2["fo_capable"], s3["fo_capable"]] == [10, 20, 30]
    assert [s["width_mean"] for s in (s1, s2, s3)] == pytest.approx(reference["width"])
    assert [s["n_eff"] for s in (s1, s2, s3)] == pytest.approx(reference["n_eff"], abs=0.006)
    assert s1["max_share"] == pytest.approx(reference["s1_max_share"]) and s1["top_share"] == pytest.approx(reference["s1_top6_share"])
    assert result["max_presence"] == pytest.approx(reference["max_presence"]) and result["bias"] == pytest.approx(reference["bias"], abs=0.0006)
    assert result["dzo_over_q"] == pytest.approx(reference["dzo_over_q"]["qwen"], abs=0.5)


@pytest.mark.parametrize("name", sorted(supply.APPENDIX_A))
def test_simulated_width_distribution_of_appendix_a(qwen, name):
    reference, s1 = supply.APPENDIX_A[name], qwen[name]["simulated"][1]
    assert s1["rounds"] == 60 * 160
    for key in ("p_s1_0", "p_s1_1", "p_s1_le2"):
        assert s1[key.replace("s1_", "width_")] == pytest.approx(reference[key], abs=0.02)
    assert s1["width_mean"] == pytest.approx(reference["width"][0], abs=0.05)


def test_uniform_sampling_follows_the_hypergeometric_distribution(qwen):
    histogram = np.asarray(qwen["dep_round_v1 beta=0"]["simulated"][1]["histogram"]) / (60 * 160)
    assert histogram == pytest.approx([0.0653, 0.2611, 0.3672, 0.2304, 0.0672, 0.0085, 0.0004], abs=0.02)


def test_quota_2_2_2_has_width_two_in_every_round(qwen):
    s1 = qwen["tier_quota 2,2,2"]["simulated"][1]
    assert (s1["p_width_0"], s1["p_width_le2"], s1["p_below_quorum"], s1["width_mean"]) == (0.0, 1.0, 1.0, 2.0)


def test_opt_differs_only_in_the_dimension_row():
    result = supply.plan("opt", seeds=2, rounds=10)
    for name, reference in supply.APPENDIX_A.items():
        assert result[name]["analytic"]["dzo_over_q"] == pytest.approx(reference["dzo_over_q"]["opt"], abs=0.5)
        assert [s["width_mean"] for s in result[name]["analytic"]["segments"][1:]] == pytest.approx(reference["width"])
    assert supply.segments(SPECS["opt"][3], 12) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert supply.segments([0] * 30, 12) == [(0, 12)]


def test_table_csv_and_command_line(qwen, tmp_path, capsys):
    text = supply.plan_table("qwen", qwen)
    assert "tier_quota 3,1,2" in text and "(brief 6.49)" in text
    rows = supply.plan_rows("qwen", qwen)
    assert {"source", "scheduler", "model", "segment", "metric", "value"} == set(rows[0]) and len(rows) > 200
    supply.main(["plan", "--model", "opt", "--seeds", "2", "--rounds", "5", "--output", str(tmp_path)])
    assert "Supply map for opt" in capsys.readouterr().out
    assert (tmp_path / "supply_opt.csv").read_text().startswith("source,scheduler,model,segment,metric,value")
    assert set(json.loads((tmp_path / "supply_opt.json").read_text())) == set(supply.E1A_SETTINGS)


def test_empirical_supply_from_a_ledger(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "opt", "k": 4}))
    ledger = [{"round": 1, "client": c, "cut": cut} for c, cut in ((0, 3), (1, 3), (10, 6), (20, 9))] \
        + [{"round": 2, "client": c, "cut": cut} for c, cut in ((10, 6), (20, 9), (21, 9), (22, 9))]
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    result = supply.empirical(tmp_path)
    s1, s2, s3 = result["segments"][1:]
    assert (s1["width_mean"], s2["width_mean"], s3["width_mean"]) == (1.0, 2.0, 4.0)
    assert (s1["p_width_0"], s1["p_width_le2"], s1["p_below_quorum"]) == (0.5, 1.0, 1.0)
    assert s1["sources"] == 2 and s1["n_eff"] == pytest.approx(2.0) and result["max_presence"] == 1.0
    assert "mean width S1/S2/S3 1.00 / 2.00 / 4.00" in supply.empirical_table([tmp_path])
