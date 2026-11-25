"""Correctness checks of the satellite code on a loaded model (brief v2, section 10.1, E0-a items 2 to 5).

The same functions run in the CPU tests on the tiny models and, on the GPU, on the real ones:

    python -m ho_fl.sat.selfcheck --task t2 --output outputs/sat/e0a/selfcheck-t2 [SatConfig options]
    python -m ho_fl.sat.selfcheck upstream --model qwen --dataset ag_news --partition dirichlet_05 \\
        --output outputs/sat/e0a/selfcheck-upstream [ho-fl-train options]

Items (numbers as in the brief):

2. ``hybrid_step`` with an explicit cut is bitwise the step with the derived cut for the cuts upstream supports;
   with cut 0 its gradient is the one of plain backpropagation; with the cut at the model depth only the
   zeroth-order path moves the transformer blocks.
3. ``train_participation`` is bitwise ``ho_fl.train.train_client`` for the tier cuts (second form of the command:
   it needs an upstream backend, i.e. a language model with an upstream dataset).
4. With an ideal network the satellite aggregation equals ``ho_fl.aggregation.aggregate`` followed by ``apply``
   with the cohort size set to the number of participants: first-order terms bitwise, zeroth-order terms within
   the replay tolerance.
5. Replaying the same reports on two replicas with the same state gives bitwise the same result.

A row is ``{"item", "check", "passed", "detail", "worst", "bitwise"}``; the last two are the largest difference the
check found and whether the compared tensors were bitwise the same (``None`` where that does not apply). Checks restore the parameters they found; items 3 and 4 advance
the data streams of the satellites they train, so run them on a backend that is not used for a run afterwards.

The first form of the command also runs a diagnostic that is not a pass/fail item (``check_estimator``): how well the
zeroth-order scalar ``v / mu`` of training follows the exact directional derivative on this model and data, per cut
and perturbation radius. The brief fixes ``mu = 1e-2`` for the satellite tasks so that the estimate is not
dominated by bfloat16 rounding (section 1.6); this measures whether that holds on the task at hand.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..aggregation import aggregate, apply
from ..algorithms import hybrid_step, restore
from ..batches import ClientStream
from ..io import write_json
from ..model import block_index
from ..randomness import RoundDirections, stable_seed
from ..replay import ATOL
from .aggregate import Groups, replay_zeroth_order, round_terms, round_updates, sum_terms
from .client import train_participation
from .network import Delivery


class _NoStep:
    """Stands in for the optimizer when only gradients are wanted."""

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def _copy(tensors):
    return {name: tensor.detach().clone() for name, tensor in tensors.items()}


def _bitwise(a, b):
    return list(a) == list(b) and all(torch.equal(a[name], b[name]) for name in a)


def _max_abs(a, b):
    return max((float((a[name].double() - b[name].double()).abs().max()) for name in a), default=0.0)


def _clear(params):
    for param in params.values():
        param.grad = None


def _row(item, check, passed, detail="", worst=None, bitwise=None):
    return {"item": item, "check": check, "passed": bool(passed), "detail": detail, "worst": worst,
            "bitwise": None if bitwise is None else bool(bitwise)}


def check_hybrid_step(b, steps=2):
    """Item 2 on a ``SatBackend``."""
    rows, state = [], _copy(b.params)
    batch = b.batch("train", b.streams[0].indices[:b.cfg.batch_size])
    for cut in [cut for cut in b.cuts if 0 < cut < b.task.depth]:
        traces = []
        for extra in ({}, {"cut": cut}):
            restore(b.params, state)
            _clear(b.params)
            optimizer, directions, trace = b.optimizer(), RoundDirections(4242), []
            for e in range(steps):
                loss, scalars = hybrid_step(b.cfg, b.wrapper, b.params, b.bottoms[cut], batch, b.tokens, optimizer, directions, e, **extra)
                trace.append((loss.detach().clone(), [scalar.clone() for scalar in scalars], _copy(b.params)))
            traces.append(trace)
        same = all(torch.equal(l1, l2) and len(s1) == len(s2) == b.cfg.q and all(torch.equal(x, y) for x, y in zip(s1, s2)) and _bitwise(p1, p2)
                   for (l1, s1, p1), (l2, s2, p2) in zip(*traces))
        worst = max(_max_abs(p1, p2) for (_, _, p1), (_, _, p2) in zip(*traces))
        rows.append(_row(2, f"cut {cut}: the step with the explicit cut is bitwise the step with the derived cut", same,
                         f"{steps} consecutive steps, largest difference of a parameter {worst:.3g}", worst, same))
    # Cut 0 against plain backpropagation through the whole model.
    restore(b.params, state)
    _clear(b.params)
    logits = b.wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=b.tokens)[:, -1].float()
    F.cross_entropy(logits, batch["labels"], reduction="mean").backward()
    expected = {name: param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param) for name, param in b.params.items()}
    _clear(b.params)
    directions = RoundDirections(4242)
    _, scalars = hybrid_step(b.cfg, b.wrapper, b.params, {}, batch, b.tokens, _NoStep(), directions, 0, cut=0)
    actual = {name: param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param) for name, param in b.params.items()}
    difference, scale = _max_abs(actual, expected), max(float(tensor.abs().max()) for tensor in expected.values())
    rows.append(_row(2, "cut 0: the gradient is the one of plain backpropagation (tolerance 1e-6) and no direction is drawn",
                     difference <= 1e-6 * max(1.0, scale) and scalars == [] and not directions.cache and scale > 0,
                     f"largest difference {difference:.3g}, largest gradient entry {scale:.3g}, bitwise {_bitwise(actual, expected)}",
                     difference, _bitwise(actual, expected)))
    # Cut at the model depth: the blocks move through the zeroth-order path only; a head above them by backpropagation.
    depth, moved, drawn = b.task.depth, {}, {}
    for zo in (True, False):
        restore(b.params, state)
        _clear(b.params)
        _, scalars = hybrid_step(b.cfg, b.wrapper, b.params, b.bottoms[depth], batch, b.tokens, b.optimizer(), RoundDirections(4242), 0, zo=zo, cut=depth)
        moved[zo] = {name for name, param in b.params.items() if not torch.equal(param, state[name])}
        drawn[zo] = len(scalars)
    blocks = {name for name in b.params if block_index(name) < depth}
    head = set(b.params) - blocks
    rows.append(_row(2, f"cut {depth}: every block is trained forward-only, nothing below the head by backpropagation",
                     moved[True] == set(b.params) and moved[False] == head and drawn == {True: b.cfg.q, False: 0},
                     f"{len(blocks)} block tensors move with the zeroth-order part and none without it; {len(head)} head tensors move in both cases"))
    restore(b.params, state)
    _clear(b.params)
    return rows


def check_upstream_participation(upstream, rounds=(1, 2)):
    """Item 3 on an upstream ``ho_fl.train.Backend`` (method hofl): the same clients, rounds and cuts through
    ``train_client`` and through ``train_participation`` on a satellite backend built around the same model."""
    from ..config import SPECS
    from ..train import train_client
    from .backend import SatBackend
    from .config import Task

    cfg = upstream.cfg
    depth, tier_cuts = SPECS[cfg.model][2], SPECS[cfg.model][3]
    task = Task("upstream", "text", cfg.model, cfg.dataset, depth, (*tier_cuts, depth), cfg.lr, 0.0)
    sat = SatBackend.from_parts(task, cfg, upstream.model, upstream.wrapper, upstream.params, upstream.tokenizer, upstream.tokens,
                                upstream.data, upstream.manifest["clients"], len(cfg.verbalizers))
    state, rows = _copy(upstream.params), []
    for cut in tier_cuts:
        cid = upstream.resource_cuts.index(cut)
        streams = {cid: ClientStream(upstream.manifest["clients"][cid], cfg.batch_size)}
        sat.streams[cid] = ClientStream(upstream.manifest["clients"][cid], cfg.batch_size)
        same, worst = True, 0.0
        for r in rounds:
            reference = train_client(upstream, state, streams, r, cid, cut)
            candidate = train_participation(sat, state, r, cid, cut)
            same &= (_bitwise(candidate.delta, reference.delta) and candidate.batches == reference.batches
                     and all(torch.equal(a, b) for a, b in zip(candidate.losses, reference.losses))
                     and all(torch.equal(a, b) for x, y in zip(candidate.scalars, reference.scalars) for a, b in zip(x, y))
                     and [len(step) for step in candidate.scalars] == [len(step) for step in reference.scalars]
                     and (candidate.cut, candidate.zo, candidate.direction_master_seed) == (reference.cut, reference.zo, reference.direction_master_seed))
            worst = max(worst, _max_abs(candidate.delta, reference.delta))
        rows.append(_row(3, f"cut {cut}: train_participation is bitwise train_client", same,
                         f"client {cid}, rounds {list(rounds)}, largest difference of an update {worst:.3g}", worst, same))
    restore(upstream.params, state)
    return rows


def ideal_delivery(reports, b):
    """The outcome of an ideal network for these reports: everything arrives, nothing is late."""
    senders = {sid for sid, report in reports.items() if b.fo_bytes(report.cut)}
    reporting = {sid for sid, report in reports.items() if report.zo}
    return Delivery(timeline=None, arrival_s={}, on_time=senders, late=[], beacons=set(reports), zo_received=reporting,
                    zo_dropped=[], flood_s=0.0, bytes={}, tx_energy_j=None, components=[])


def train_cohort(b, state, cohort, r=1):
    """Reports of ``{sid: cut}`` trained from ``state`` in round ``r``."""
    return {sid: train_participation(b, state, r, sid, cut) for sid, cut in sorted(cohort.items())}


def default_cohort(b, with_forward_only=True):
    """Two backbone satellites, three hybrid ones at the three hybrid cuts and, optionally, a forward-only one."""
    idle, imaging, third = b.task.ladder[:3]
    cohort = {0: 0, 2: idle, 4: imaging, 6: third, 10: 0}
    if with_forward_only:
        cohort[1] = b.task.depth
    return cohort


def check_aggregation(b, server_lr=3.0):
    """Item 4 on a ``SatBackend``."""
    rows, state = [], _copy(b.params)
    for with_forward_only in (True, False):
        reports = train_cohort(b, state, default_cohort(b, with_forward_only))
        ordered = [reports[sid] for sid in sorted(reports)]
        delivery = ideal_delivery(reports, b)
        terms = round_terms(state, reports, delivery, b.cfg)
        update = sum_terms(state, terms)
        ours = _copy(state)
        apply(ours, update, server_lr)
        reference_update, _ = aggregate(ordered, None, replace(b.cfg, k=len(ordered)))
        central = _copy(state)
        apply(central, reference_update, server_lr)
        label = "two backbone, three hybrid" + (", one forward-only" if with_forward_only else "") + " satellites"
        first = all(term.first_order[name] is reports[term.sid].delta[name] or torch.equal(term.first_order[name], reports[term.sid].delta[name])
                    for term in terms for name in term.first_order)
        covered = all(set(term.first_order) | set(term.zeroth_order) == set(state) and not set(term.first_order) & set(term.zeroth_order)
                      and all((block_index(name) >= reports[term.sid].cut) == (name in term.first_order) for name in state)
                      for term in terms)
        rows.append(_row(4, f"{label}: first-order terms are the participants' own tensors, bitwise", first and covered,
                         f"{sum(len(term.first_order) for term in terms)} tensors"))
        local = {term.sid: {name: reports[term.sid].delta[name] for name in term.zeroth_order} for term in terms}
        worst = max((_max_abs(term.zeroth_order, local[term.sid]) for term in terms), default=0.0)
        exact = all(_bitwise(term.zeroth_order, local[term.sid]) for term in terms)
        rows.append(_row(4, f"{label}: replayed zeroth-order terms equal the locally computed ones within {ATOL:g}", worst <= ATOL,
                         f"largest difference {worst:.3g}, bitwise {exact}, {sum(len(term.zeroth_order) for term in terms)} tensors", worst, exact))
        lowest = max(report.cut for report in ordered if report.zo) if any(report.zo for report in ordered) else 0
        pure = [name for name in state if block_index(name) >= lowest]
        rows.append(_row(4, f"{label}: tensors that only receive first-order terms end bitwise as after central aggregation",
                         all(torch.equal(ours[name], central[name]) for name in pure) and update.divisor == reference_update.divisor,
                         f"{len(pure)} tensors (blocks from {lowest} on" + ("" if pure else "; none in this cohort") + f"), divisor {len(ordered)}"))
        difference = _max_abs(ours, central)
        rows.append(_row(4, f"{label}: the new state equals central aggregate + apply within {ATOL:g}", difference <= ATOL,
                         f"largest difference {difference:.3g}, bitwise {_bitwise(ours, central)}, server step {server_lr:g}",
                         difference, _bitwise(ours, central)))
    restore(b.params, state)
    return rows


def check_replay(b, server_lr=3.0):
    """Item 5 on a ``SatBackend``."""
    state = _copy(b.params)
    reports = train_cohort(b, state, default_cohort(b), r=2)
    delivery = ideal_delivery(reports, b)
    replicas = [_copy(state), _copy(state)]
    rebuilt = [{sid: replay_zeroth_order(replica, report, b.cfg) for sid, report in reports.items() if report.zo} for replica in replicas]
    again = {sid: replay_zeroth_order(replicas[0], report, b.cfg) for sid, report in reports.items() if report.zo}
    worst = max((max(_max_abs(rebuilt[0][sid], rebuilt[1][sid]), _max_abs(rebuilt[0][sid], again[sid])) for sid in again), default=0.0)
    same = all(_bitwise(rebuilt[0][sid], rebuilt[1][sid]) for sid in rebuilt[0]) and all(_bitwise(rebuilt[0][sid], again[sid]) for sid in again)
    rows = [_row(5, "replaying the same reports on two replicas with the same state is bitwise the same", same,
                 f"{len(rebuilt[0])} reports, also when repeated on one replica; largest difference {worst:.3g}", worst, same)]
    # Two replica groups that start equal stay equal through a round, and can then be joined again.
    groups = Groups(_copy(state), planes=4)
    first, second = groups.split(0, [[0, 1], [2, 3]])
    groups.states[second] = _copy(state)
    before = groups.state(first)
    updates, _ = round_updates(groups, reports, delivery, b.cfg)
    groups.apply(updates, server_lr)
    equal = _bitwise(groups.state(first), groups.state(second))
    apart = _max_abs(groups.state(first), groups.state(second))
    untouched = _bitwise(before, state) and before is not groups.state(first)
    groups.merge_identical()
    rows.append(_row(5, "two replica groups with the same state are bitwise equal after a round; updates do not write in place",
                     equal and untouched and groups.ids == [first] and groups.group_of == [first] * 4,
                     f"{len(reports)} participants, {sum(report.zo for report in reports.values())} with a zeroth-order part; "
                     f"largest difference between the groups {apart:.3g}", apart, equal))
    restore(b.params, state)
    return rows


ESTIMATOR_RADII = (1e-3, 1e-2, 1e-1)


def _flat(tensors, names):
    return torch.cat([tensors[name].detach().double().reshape(-1) for name in names])


def check_estimator(b, radii=ESTIMATOR_RADII, directions=32, batches=2):
    """Diagnostic: the zeroth-order scalars of ``hybrid_step`` against exact directional derivatives.

    For every cut with a forward-only part, ``batches`` training batches and ``directions`` random directions each:
    ``v / mu`` as training computes it (``hybrid_step`` with an optimizer that does nothing) and ``<gradient, u>``
    from backpropagation through the whole model. A faithful finite difference gives correlation 1 and slope 1.
    Too small a radius drowns the activation difference in bfloat16 rounding, too large a one leaves the linear
    regime; the row of the configured radius is marked. Nothing is trained and no data stream is advanced.
    """
    state, rows = _copy(b.params), []
    examples = b.streams[0].indices
    for cut in [cut for cut in b.cuts if cut > 0]:
        prefix = b.bottoms[cut]
        names = list(prefix)
        exact, estimates = [], {mu: [] for mu in radii}
        for step in range(batches):
            batch = b.batch("train", examples[step * b.cfg.batch_size:(step + 1) * b.cfg.batch_size])
            restore(b.params, state)
            _clear(b.params)
            logits = b.wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=b.tokens)[:, -1].float()
            F.cross_entropy(logits, batch["labels"], reduction="mean").backward()
            gradient = _flat({name: b.params[name].grad if b.params[name].grad is not None else torch.zeros_like(b.params[name])
                              for name in names}, names)
            drawn = RoundDirections(stable_seed("sat-estimator", cut, step))
            exact += [float(gradient @ _flat(drawn.get(prefix, step, p), names)) for p in range(directions)]
            for mu in radii:
                restore(b.params, state)
                _clear(b.params)
                _, scalars = hybrid_step(SimpleNamespace(q=directions, mu=mu), b.wrapper, b.params, prefix, batch, b.tokens, _NoStep(),
                                         drawn, step, cut=cut)
                estimates[mu] += [float(scalar) / mu for scalar in scalars]
            drawn.cache.clear()
        truth = np.array(exact)
        for mu in radii:
            estimate = np.array(estimates[mu])
            finite = bool(np.isfinite(estimate).all())
            varied = finite and truth.std() > 0 and estimate.std() > 0
            rows.append({"cut": cut, "mu": mu, "configured": mu == b.cfg.mu, "directions": int(truth.size), "dimension": int(gradient.numel()),
                         "finite": finite,
                         "correlation": float(np.corrcoef(estimate, truth)[0, 1]) if varied else None,
                         "slope": float((estimate * truth).sum() / (truth * truth).sum()) if finite and (truth * truth).sum() > 0 else None,
                         "sign_agreement": float(np.mean(np.sign(estimate) == np.sign(truth))) if finite else None})
    restore(b.params, state)
    _clear(b.params)
    return rows


def render_estimator(rows, configured):
    show = lambda value: "n/a" if value is None else f"{value:.3f}"  # noqa: E731
    lines = ["Zeroth-order scalars against exact directional derivatives (diagnostic, not a pass/fail item; 1 and 1 mean the finite "
             f"difference is exact; the runs use mu = {configured:g}):", "",
             "| cut | mu | directions | correlation | slope | same sign |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {row['cut']} | {row['mu']:g}{' (configured)' if row['configured'] else ''} | {row['directions']} | {show(row['correlation'])} | "
              f"{show(row['slope'])} | {show(row['sign_agreement'])} |" for row in rows]
    return "\n".join(lines) + "\n"


def render(rows, title):
    lines = [title, "", "| item | check | result | detail |", "| --- | --- | --- | --- |"]
    lines += [f"| {row['item']} | {row['check']} | {'pass' if row['passed'] else 'FAIL'} | {row['detail']} |" for row in rows]
    failed = sum(not row["passed"] for row in rows)
    lines += ["", f"{len(rows) - failed} of {len(rows)} checks passed" + (f"; {failed} FAILED" if failed else "")]
    return "\n".join(lines) + "\n"


def finish(rows, out, title, extra):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    passed = all(row["passed"] for row in rows)
    write_json(out / "selfcheck.json", {**extra, "status": "PASSED" if passed else "FAILED", "rows": rows})
    (out / "selfcheck.md").write_text(render(rows, title))
    print(render(rows, title), flush=True)
    print(json.dumps({"status": "PASSED" if passed else "FAILED", "checks": len(rows)}), flush=True)
    return 0 if passed else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    environment = {"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if argv and argv[0] == "upstream":
        from ..config import parse_config
        from ..train import Backend

        cfg = parse_config(argv[1:])
        print(f"> loading {cfg.model} and {cfg.dataset} as upstream HO-FL does", flush=True)
        upstream = Backend(cfg)
        print("> item 3: one participation through upstream train_client and through the satellite code, three cuts, two rounds each", flush=True)
        rows = check_upstream_participation(upstream)
        return finish(rows, cfg.output, f"Self-check against upstream HO-FL ({cfg.model}, {cfg.dataset}, {cfg.device})",
                      {**environment, "model": cfg.model, "dataset": cfg.dataset, "device": cfg.device})
    from .backend import SatBackend
    from .config import parse_sat_config

    sat = parse_sat_config(argv)
    print(f"> loading the model and the data of task {sat.task} ({sat.spec.model}, {sat.spec.dataset})", flush=True)
    b = SatBackend(sat)
    rows = []
    for item, what, check in ((2, "local steps with explicit cuts", check_hybrid_step), (4, "aggregation against the central rule", check_aggregation),
                              (5, "replay on two replicas", check_replay)):
        print(f"> item {item}: {what}", flush=True)
        found = check(b)
        print(f">   {sum(row['passed'] for row in found)} of {len(found)} checks passed", flush=True)
        rows += found
    print("> diagnostic: zeroth-order scalars against exact directional derivatives, every cut, three radii", flush=True)
    estimator = check_estimator(b)
    print(render_estimator(estimator, b.cfg.mu), flush=True)
    return finish(rows, sat.output, f"Self-check of task {sat.task} ({b.task.model}, {b.task.dataset}, {sat.device}{', tiny assets' if sat.tiny else ''})",
                  {**environment, "task": sat.task, "model": b.task.model, "device": sat.device, "tiny": sat.tiny, "mu": b.cfg.mu,
                   "prompt_tokens": b.token_meta, "estimator": estimator})


if __name__ == "__main__":
    sys.exit(main())
