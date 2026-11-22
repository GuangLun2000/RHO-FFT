"""Numerical check of the zeroth-order part of HO-FL (no training).

HO-FL estimates the gradient of the bottom segment from scalars ``v = <lambda, h(theta + mu u) - h(theta)>``, where
``lambda`` is the gradient of the loss at the cut, ``h`` the activations at the cut and ``u`` a random direction.
To first order ``v / mu`` is the directional derivative ``<grad, u>`` of the loss. This module computes both for the
same batches and directions:

- ``v`` exactly as training does: by calling ``hybrid_step`` itself with an optimizer that does nothing, on the
  directions training would draw for that client;
- the exact directional derivative from autograd, through the full model.

Reported per state, precision of the backbone (bfloat16 as in training, float32 as a reference), perturbation radius
``mu`` and cut:

- fidelity: correlation and regression slope of ``v / mu`` against the exact derivative (1 and 1 if the finite
  difference is accurate), sign agreement and relative error;
- information: cosine between the gradient and the q-direction estimate ``(1/q) sum_p u_p v_p / mu`` for growing q,
  next to the same estimate built from exact derivatives and the theory ``sqrt(q / d)``.

    python -m ho_fl.zo_check --output DIR [--states ",path/to/state.safetensors"] [--q 64] [--mus 1e-4,1e-3,1e-2]
                             [--batches 2] [--precisions bfloat16,float32] [ho-fl-train options, e.g. --model qwen ...]
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .algorithms import hybrid_step, restore, task_loss
from .batches import ClientStream, collate
from .config import parse_config
from .io import write_json
from .model import partition_parameters
from .randomness import client_directions
from .states import load_state

Q_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class _NoStep:
    """Stands in for the optimizer: hybrid_step computes its scalars, nothing is updated."""

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def _flat(tensors, names):
    return torch.cat([tensors[name].detach().double().reshape(-1) for name in names])


def exact_gradient(b, state, batch, names):
    """Gradient of the batch loss for the tensors in ``names`` by backpropagation through the whole model."""
    restore(b.params, state)
    for param in b.params.values():
        param.grad = None
    task_loss(b.wrapper, batch, b.tokens).backward()
    gradient = {name: b.params[name].grad.detach().clone() if b.params[name].grad is not None else torch.zeros_like(b.params[name]) for name in names}
    for param in b.params.values():
        param.grad = None
    return gradient


def client_check(b, state, cid, cut, q, mus, batches, round_id=1):
    """Exact derivatives and HO-FL scalars of one client at ``state`` for ``batches`` local steps; one row per mu."""
    cfg = b.cfg
    prefix, _ = partition_parameters(b.params, cut)
    names = list(prefix)
    stream = ClientStream(b.manifest["clients"][cid], cfg.batch_size)
    directions = client_directions(cfg.seed, cfg.model, round_id, cid)
    samples = []
    for step in range(batches):
        batch = collate(b.data["train"], stream.next(), b.tokenizer.pad_token_id, cfg.device)
        gradient = _flat(exact_gradient(b, state, batch, names), names)
        units = [_flat(directions.get(prefix, step, p), names) for p in range(q)]
        exact = np.array([float(gradient @ u) for u in units])
        estimated = {}
        for mu in mus:
            restore(b.params, state)
            _, scalars = hybrid_step(SimpleNamespace(q=q, mu=mu), b.wrapper, b.params, prefix, batch, b.tokens, _NoStep(), directions, step)
            estimated[mu] = np.array([float(v) / mu for v in scalars])
            for param in b.params.values():
                param.grad = None
        samples.append((gradient, units, exact, estimated))
        directions.cache.clear()
    restore(b.params, state)
    dimension = int(samples[0][0].numel())
    rows = []
    for mu in mus:
        exact = np.concatenate([sample[2] for sample in samples])
        estimate = np.concatenate([sample[3][mu] for sample in samples])
        correlation = float(np.corrcoef(estimate, exact)[0, 1]) if exact.std() > 0 and estimate.std() > 0 else None
        slope = float((estimate * exact).sum() / (exact * exact).sum()) if (exact * exact).sum() > 0 else None
        relative = float(np.median(np.abs(estimate - exact)) / np.median(np.abs(exact))) if np.median(np.abs(exact)) > 0 else None
        information = []
        for count in (value for value in Q_STEPS if value <= q):
            cos_hofl, cos_exact = [], []
            for gradient, units, derivatives, by_mu in samples:
                for values, target in ((by_mu[mu], cos_hofl), (derivatives, cos_exact)):
                    estimate_vector = sum(units[p] * values[p] for p in range(count)) / count
                    denominator = float(estimate_vector.norm() * gradient.norm())
                    target.append(float(estimate_vector @ gradient) / denominator if denominator > 0 else 0.0)
            information.append({"q": count, "cosine_hofl": float(np.mean(cos_hofl)), "cosine_exact_derivatives": float(np.mean(cos_exact)),
                                "theory": math.sqrt(count / (dimension + count + 1))})
        rows.append({"client": cid, "cut": cut, "mu": mu, "directions": int(exact.size), "dimension": dimension,
                     "correlation": correlation, "slope": slope, "sign_agreement": float(np.mean(np.sign(estimate) == np.sign(exact))),
                     "median_relative_error": relative, "information": information})
    return rows


def run(cfg, out, states=("",), q=64, mus=(1e-4, 1e-3, 1e-2), batches=2, precisions=("bfloat16", "float32")):
    """Rows for every state, precision, mu and tier cut; written to ``out/zo_check.json`` with a short table."""
    from .train import Backend

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = Backend(cfg)
    initial = {name: param.detach().clone() for name, param in b.params.items()}
    loaded = {"": ("initial", initial)}
    for path in states:
        if path:
            state, record = load_state(path, initial)
            loaded[path] = (f"round {record['metadata'].get('round', '?')}", state)
    clients = {cut: b.resource_cuts.index(cut) for cut in sorted(set(b.resource_cuts))}
    rows = []
    for precision in precisions:
        if precision == "float32":
            b.model.to(torch.float32)  # an exact float32 copy of the bfloat16 backbone; the adapters already are float32
        elif precision != "bfloat16":
            raise ValueError(f"Unknown precision {precision}")
        for path in states:
            label, state = loaded[path]
            for cut, cid in clients.items():
                for row in client_check(b, state, cid, cut, q, mus, batches):
                    rows.append({"state": label, "precision": precision, **row})
                    show = lambda value, digits: "n/a" if value is None else f"{value:.{digits}f}"  # noqa: E731
                    print(f"    estimator check | state {label} | backbone {precision} | mu {row['mu']:g} | cut {cut} (client {cid}): "
                          f"correlation with the exact derivative {show(row['correlation'], 4)}, slope {show(row['slope'], 3)}, "
                          f"same sign {show(row['sign_agreement'], 3)}", flush=True)
    result = {"model": cfg.model, "dataset": cfg.dataset, "seed": cfg.seed, "q": q, "batches": batches, "mus": list(mus),
              "torch_version": torch.__version__, "device": cfg.device, "rows": rows}
    write_json(out / "zo_check.json", result)
    (out / "zo_check.md").write_text(render(result))
    print("\nEstimator check (correlation and slope 1 mean HO-FL's finite difference equals the exact directional derivative; "
          "the last column compares the q = 2 estimate's cosine with the gradient: HO-FL / from exact derivatives / theory):\n", flush=True)
    print(render(result), flush=True)
    return result


def render(result):
    lines = ["| state | backbone | mu | cut | directions | correlation | slope | sign agreement | median relative error | cosine q=2 (HO-FL / exact / theory) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    show = lambda value, digits=3: "n/a" if value is None else f"{value:.{digits}g}"  # noqa: E731
    for row in result["rows"]:
        at2 = next((entry for entry in row["information"] if entry["q"] == 2), None)
        cosine = "n/a" if at2 is None else f"{at2['cosine_hofl']:.4f} / {at2['cosine_exact_derivatives']:.4f} / {at2['theory']:.4f}"
        lines.append(f"| {row['state']} | {row['precision']} | {row['mu']:g} | {row['cut']} | {row['directions']} | {show(row['correlation'])} | "
                     f"{show(row['slope'])} | {show(row['sign_agreement'])} | {show(row['median_relative_error'])} | {cosine} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", default="", help="comma-separated saved states to check besides the initial one ('' = initial)")
    parser.add_argument("--q", type=int, default=64, help="directions per batch")
    parser.add_argument("--mus", default="1e-4,1e-3,1e-2")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--precisions", default="bfloat16,float32")
    parser.add_argument("--tiny", action="store_true", help="CPU dry run on the synthetic tiny assets (the numbers mean nothing)")
    args, rest = parser.parse_known_args(argv)
    cfg = parse_config(rest)
    if args.tiny:
        from .config import Config
        from .testing import build_tiny_assets

        assets = build_tiny_assets(Path(cfg.output) / "assets", datasets=("ag_news",))["ag_news"]
        cfg = Config(**{**assets, "partition": "dirichlet_05", "seed": cfg.seed, "output": cfg.output}).resolve()
    states = [""] + [part for part in args.states.split(",") if part]
    run(cfg, cfg.output, states=states, q=args.q, mus=[float(part) for part in args.mus.split(",")], batches=args.batches,
        precisions=[part for part in args.precisions.split(",") if part])


if __name__ == "__main__":
    sys.exit(main())
