"""HumAID and EuroSAT for the satellite scenario, and the two-level partition over planes and satellites
(brief v2, section 7).

Data. HumAID is read from the per-event release ``QCRI/HumAID-events``: the pooled release named in the brief has
no event field. The 19 event subsets, taken in name order, add up to the same three splits (53,531 / 7,793 /
15,160) and carry 10 class labels. EuroSAT RGB is read from ``timm/eurosat-rgb`` and kept as uint8 arrays. Both
are cached under ``<cache_dir>/sat/data`` with a hash of their content.

Partition. Level one gives every plane the same number of training examples with its own skew: for HumAID 70% of
a plane come from its "main" events (event ``e`` in name order belongs to plane ``e mod P``), for EuroSAT every
plane draws a class prior from Dirichlet(0.5). Level two splits a plane among its satellites: each satellite draws
a class distribution from Dirichlet(5 x the plane's class distribution) and takes an equal share. In the "zonly"
variant two events (HumAID) or one class (EuroSAT) are held out and their training examples go to the forward-only
satellites alone.

All random draws come from generators seeded with ``stable_seed``; nothing uses a global generator.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from ..io import write_json
from ..randomness import stable_seed
from .config import SOURCES
from .roles import Roles

HUMAID_EVENTS = (
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019", "ecuador_earthquake_2016",
    "greece_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "hurricane_matthew_2016", "italy_earthquake_aug_2016",
    "kaikoura_earthquake_2016", "kerala_floods_2018", "maryland_floods_2018", "midwestern_us_floods_2019",
    "pakistan_earthquake_2019", "puebla_mexico_earthquake_2017", "srilanka_floods_2017")
HUMAID_SPLITS = {"train": "train", "validation": "dev", "test": "test"}
HUMAID_COUNTS = {"train": 53531, "validation": 7793, "test": 15160}
EUROSAT_COUNTS = {"train": 16200, "validation": 5400, "test": 5400}
EUROSAT_CLASSES = ("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",
                   "Residential", "River", "SeaLake")
SPLITS = ("train", "validation", "test")
MAIN_SHARE = (7, 10)          # share of a plane's examples that come from its main events (HumAID)
PLANE_ALPHA = 0.5             # Dirichlet concentration of a plane's class prior (EuroSAT)
SATELLITE_ALPHA = 5.0         # Dirichlet concentration of a satellite's class distribution around its plane's
ZONLY_EVENTS = 2              # HumAID: the last events in name order are held out
ZONLY_CLASS = 8               # EuroSAT: the class with this number in the dataset's label order is held out
T1_EVAL_SUBSET = 4000         # HumAID curves use a fixed subset of the test split of this size
VERBALIZER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def data_dir(cache_dir, dataset, tiny=False):
    return Path(cache_dir) / "sat" / "data" / (dataset + ("-tiny" if tiny else ""))


def _content_hash(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else json.dumps(part, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def labels_hash(data):
    """Hash of what a partition depends on: the classes, the labels of every split and, for HumAID, the events.
    Unlike the content hash of a dataset it does not involve pixels or texts, so it is the same on every machine
    (EuroSAT is stored as JPEG, and decoded pixels may differ by a unit between decoder versions)."""
    return _content_hash([data["classes"], data.get("events", [])]
                         + [list(data[split][key]) for split in SPLITS for key in ("labels", "events") if key in data[split]])


def partition_hash(manifest):
    """Hash of a partition proper: which training examples every satellite holds and what is held out. Unlike
    the hash of the manifest file it covers nothing that names a library version or a decoded pixel, so two
    machines that draw the same partition report the same value."""
    held = manifest["held_out"] or {}
    return _content_hash([manifest["task"], manifest["variant"], manifest["seed"], manifest["planes"], manifest["per_plane"],
                          [[record["sid"], record["data_seed"], record["indices"]] for record in manifest["satellites"]],
                          {key: held.get(key) for key in ("ids", "test_indices", "validation_indices")}])


# --- HumAID --------------------------------------------------------------------------------------------------------------

def humaid_hash(data):
    """Hash of everything the partition and the tokenisation depend on: classes, events, texts, labels, event ids."""
    return _content_hash([data["classes"], data["events"]] + [data[split][key] for split in SPLITS for key in ("text", "labels", "events")])


def download_humaid(cache_dir, local_files_only=False):
    """Read the 57 per-event files of the pinned revision and concatenate them in event name order."""
    from huggingface_hub import hf_hub_download

    identifier, revision = SOURCES["humaid"]
    rows = {split: [] for split in SPLITS}
    for event_id, event in enumerate(HUMAID_EVENTS):
        for split, name in HUMAID_SPLITS.items():
            path = hf_hub_download(identifier, f"{event}/{name}.json", repo_type="dataset", revision=revision,
                                   cache_dir=cache_dir, local_files_only=local_files_only)
            for row in json.loads(Path(path).read_text(encoding="utf-8")):
                rows[split].append((event_id, row["tweet_text"], row["class_label"]))
    classes = sorted({label for split in SPLITS for _, _, label in rows[split]})
    data = {"classes": classes, "events": list(HUMAID_EVENTS)}
    for split in SPLITS:
        data[split] = {"text": [text for _, text, _ in rows[split]], "labels": [classes.index(label) for _, _, label in rows[split]],
                       "events": [event_id for event_id, _, _ in rows[split]]}
    assert {split: len(data[split]["labels"]) for split in SPLITS} == HUMAID_COUNTS
    data["meta"] = {"source": identifier, "revision": revision, "sha256": humaid_hash(data),
                    "note": "per-event release concatenated in event name order; QCRI/HumAID-all has no event field"}
    return data


def load_humaid(cache_dir, tiny=False, local_files_only=False):
    """``{"classes", "events", "train" | "validation" | "test": {"text", "labels", "events"}, "meta"}``."""
    path = data_dir(cache_dir, "humaid", tiny) / "data.json"
    if not path.exists():
        if tiny:
            raise FileNotFoundError(f"{path} is missing; build the tiny assets with ho_fl.sat.testing.build_tiny_assets")
        write_json(path, download_humaid(cache_dir, local_files_only))
    data = json.loads(path.read_text())
    if humaid_hash(data) != data["meta"]["sha256"]:
        raise ValueError(f"{path} does not match its recorded hash")
    return data


def humaid_prompt(text, classes):
    """The AG News prompt of upstream HO-FL with HumAID's classes: letters as options, the answer at the end."""
    options = ", ".join(f"{letter} {name.replace('_', ' ')}" for letter, name in zip(VERBALIZER_LETTERS, classes))
    return f"Tweet: {text}\nCategory ({options}):"


def verbalizers(class_count):
    return [" " + letter for letter in VERBALIZER_LETTERS[:class_count]]


def tokenize_humaid(data, tokenizer, max_length):
    """Token ids of the prompts of all three splits; truncation removes the start of a tweet, never the options."""
    result = {}
    for split in SPLITS:
        prompts = [humaid_prompt(text, data["classes"]) for text in data[split]["text"]]
        ids = tokenizer(prompts, truncation=True, max_length=max_length, padding=False, add_special_tokens=True)["input_ids"]
        result[split] = {"input_ids": ids, "labels": list(data[split]["labels"])}
    return result


def load_humaid_tokens(cache_dir, data, tokenizer, model, max_length, tiny=False):
    """Tokenised HumAID, cached per model under ``<cache_dir>/sat/tokens`` and tied to the data by its hash."""
    path = Path(cache_dir) / "sat" / "tokens" / f"humaid{'-tiny' if tiny else ''}-{model}-len{max_length}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["meta"]["data_sha256"] == data["meta"]["sha256"]:
            return cached
    tokens = tokenize_humaid(data, tokenizer, max_length)
    tokens["meta"] = {"data_sha256": data["meta"]["sha256"], "model": model, "max_length": max_length,
                      "prompt": humaid_prompt("{text}", data["classes"]), "truncation_side": tokenizer.truncation_side,
                      "longest": max(len(ids) for split in SPLITS for ids in tokens[split]["input_ids"]),
                      "truncated": sum(len(ids) >= max_length for split in SPLITS for ids in tokens[split]["input_ids"])}
    write_json(path, tokens)
    return tokens


# --- EuroSAT -------------------------------------------------------------------------------------------------------------

def download_eurosat(cache_dir, local_files_only=False):
    """Decode the three parquet files of the pinned revision into uint8 arrays ``N x 64 x 64 x 3``."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download
    from PIL import Image

    identifier, revision = SOURCES["eurosat"]
    arrays = {}
    for split in SPLITS:
        path = hf_hub_download(identifier, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=revision,
                               cache_dir=cache_dir, local_files_only=local_files_only)
        table = parquet.read_table(path, columns=["image", "label"]).to_pydict()
        images = np.stack([np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"), dtype=np.uint8) for item in table["image"]])
        arrays[split] = (images, np.asarray(table["label"], dtype=np.int64))
        assert images.shape == (EUROSAT_COUNTS[split], 64, 64, 3), images.shape
    return arrays, {"source": identifier, "revision": revision, "classes": list(EUROSAT_CLASSES)}


def eurosat_hash(arrays, classes):
    parts = [list(classes)]
    for split in SPLITS:
        images, labels = arrays[split]
        parts += [np.ascontiguousarray(images).tobytes(), np.ascontiguousarray(labels.astype(np.int64)).tobytes()]
    return _content_hash(parts)


def load_eurosat(cache_dir, tiny=False, local_files_only=False):
    """``{"classes", "train" | "validation" | "test": {"images": uint8 N x H x W x 3, "labels": int list}, "meta"}``."""
    root = data_dir(cache_dir, "eurosat", tiny)
    if not (root / "meta.json").exists():
        if tiny:
            raise FileNotFoundError(f"{root} is missing; build the tiny assets with ho_fl.sat.testing.build_tiny_assets")
        arrays, meta = download_eurosat(cache_dir, local_files_only)
        write_eurosat(root, arrays, meta)
    meta = json.loads((root / "meta.json").read_text())
    arrays = {}
    for split in SPLITS:
        with np.load(root / f"{split}.npz") as stored:
            arrays[split] = (stored["images"], stored["labels"])
    if eurosat_hash(arrays, meta["classes"]) != meta["sha256"]:
        raise ValueError(f"{root} does not match its recorded hash")
    data = {"classes": meta["classes"], "meta": meta}
    for split in SPLITS:
        data[split] = {"images": arrays[split][0], "labels": arrays[split][1].tolist()}
    return data


def write_eurosat(root, arrays, meta):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        np.savez_compressed(root / f"{split}.npz", images=arrays[split][0], labels=arrays[split][1])
    write_json(root / "meta.json", {**meta, "sha256": eurosat_hash(arrays, meta["classes"]),
                                    "counts": {split: int(len(arrays[split][1])) for split in SPLITS},
                                    "image_shape": list(arrays["train"][0].shape[1:])})


def load_task_data(cfg):
    """The raw data of the configuration's task."""
    loader = load_humaid if cfg.spec.dataset == "humaid" else load_eurosat
    return loader(cfg.cache_dir, tiny=cfg.tiny, local_files_only=cfg.local_files_only)


def fixed_subset(count, size, name):
    """``size`` indices out of ``count``, sorted; decided by ``name`` alone, so the same for every seed and method."""
    if size >= count:
        return list(range(count))
    return sorted(np.random.default_rng(stable_seed(name)).permutation(count)[:size].tolist())


# --- partition -----------------------------------------------------------------------------------------------------------

def _rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def dirichlet(rng, alpha):
    """A Dirichlet draw from gamma variates; classes with zero concentration get zero mass."""
    alpha = np.asarray(alpha, dtype=np.float64)
    draw = np.zeros_like(alpha)
    positive = alpha > 0
    draw[positive] = rng.standard_gamma(alpha[positive])
    total = draw.sum()
    return draw / total if total > 0 else positive / positive.sum()


def largest_remainder(weights, total):
    """Non-negative integers proportional to ``weights`` that sum to ``total`` (ties go to the lower index)."""
    weights = np.asarray(weights, dtype=np.float64)
    counts = np.zeros(len(weights), dtype=np.int64)
    if total == 0:
        return counts
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(np.int64)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:total - counts.sum()]] += 1
    return counts


def allocate_counts(weights, total, stock):
    """Counts per class that sum to ``total``, follow ``weights`` and never exceed ``stock``. What a class cannot
    supply is redistributed over the classes that still can, in proportion to their weights (or to their stock
    when no weight is left there)."""
    weights, stock = np.asarray(weights, dtype=np.float64), np.asarray(stock, dtype=np.int64)
    if stock.sum() < total:
        raise ValueError(f"Cannot take {total} examples from a stock of {int(stock.sum())}")
    counts = np.zeros(len(stock), dtype=np.int64)
    while counts.sum() < total:
        left = stock - counts
        share = np.where(left > 0, weights, 0.0)
        if share.sum() <= 0:
            share = left.astype(np.float64)
        counts += np.minimum(largest_remainder(share, total - counts.sum()), left)
    return counts


def _class_pools(indices, labels, classes, seed_parts):
    """Per class, the given example indices in a seeded random order."""
    indices = np.asarray(indices, dtype=np.int64)
    pools = []
    for c in range(classes):
        members = indices[labels[indices] == c]
        pools.append(members[_rng(*seed_parts, c).permutation(len(members))])
    return pools


def _take(pools, cursor, counts):
    taken = []
    for c, count in enumerate(counts):
        taken.extend(pools[c][cursor[c]:cursor[c] + count].tolist())
        cursor[c] += count
    return taken


def split_among_satellites(indices, labels, classes, sizes, seed_parts):
    """Level two: split the examples of one plane into parts of the given sizes, one class distribution per part
    drawn from Dirichlet(5 x the plane's class distribution). Returns the parts and the drawn distributions."""
    pools = _class_pools(indices, labels, classes, (*seed_parts, "pool"))
    stock = np.array([len(pool) for pool in pools])
    prior = stock / stock.sum()
    cursor, parts, drawn = np.zeros(classes, dtype=np.int64), [], []
    for position, size in enumerate(sizes):
        target = dirichlet(_rng(*seed_parts, "prior", position), SATELLITE_ALPHA * prior)
        counts = allocate_counts(target, size, stock - cursor)
        parts.append(_take(pools, cursor, counts))
        drawn.append(target.tolist())
    return parts, drawn


def humaid_planes(labels, events, usable, planes, per_plane_count, seed):
    """Level one for HumAID. Every plane first takes up to 70% of its examples from its main events (all of them
    if there are fewer), then the planes fill up, in order, from one random order of everything that is left."""
    names = sorted(set(events[usable].tolist()))
    main_quota = MAIN_SHARE[0] * per_plane_count // MAIN_SHARE[1]
    chosen, summary, used = [], [], np.zeros(len(labels), dtype=bool)
    for plane in range(planes):
        mine = [event for position, event in enumerate(names) if position % planes == plane]
        pool = usable[np.isin(events[usable], mine)]
        picked = pool[_rng(seed, "sat-partition", "humaid", "main", plane).permutation(len(pool))][:main_quota]
        used[picked] = True
        chosen.append(picked.tolist())
        summary.append({"plane": plane, "main_events": mine, "main_available": int(len(pool)), "main_taken": int(len(picked))})
    rest = usable[~used[usable]]
    rest = rest[_rng(seed, "sat-partition", "humaid", "fill").permutation(len(rest))]
    cursor = 0
    for plane in range(planes):
        need = per_plane_count - len(chosen[plane])
        chosen[plane].extend(rest[cursor:cursor + need].tolist())
        cursor += need
    return chosen, summary, rest[cursor:].tolist()


def eurosat_planes(labels, usable, classes, planes, per_plane_count, seed):
    """Level one for EuroSAT. Planes take their examples in order, each by a class prior from Dirichlet(0.5) over
    the classes present; a class that runs out is made up from the others."""
    pools = _class_pools(usable, labels, classes, (seed, "sat-partition", "eurosat", "pool"))
    stock = np.array([len(pool) for pool in pools])
    cursor, chosen, summary = np.zeros(classes, dtype=np.int64), [], []
    for plane in range(planes):
        prior = dirichlet(_rng(seed, "sat-partition", "eurosat", "prior", plane), PLANE_ALPHA * (stock > 0))
        counts = allocate_counts(prior, per_plane_count, stock - cursor)
        chosen.append(_take(pools, cursor, counts))
        summary.append({"plane": plane, "class_prior": prior.tolist()})
    left = [index for c in range(classes) for index in pools[c][cursor[c]:].tolist()]
    return chosen, summary, left


def build_manifest(task, data, seed, variant="base", planes=12, per_plane=10):
    """The partition of ``data`` (the output of a loader) for ``task`` (a ``config.Task``): one record per satellite."""
    if variant not in ("base", "zonly"):
        raise ValueError(f"Unknown data variant {variant}")
    roles = Roles(planes, per_plane)
    labels = np.asarray(data["train"]["labels"], dtype=np.int64)
    classes, total = len(data["classes"]), len(labels)
    forward = [slot for slot in range(per_plane) if roles.kind(slot) == "Z"]
    held, held_out = np.zeros(total, dtype=bool), None
    if variant == "zonly":
        if task.dataset == "humaid":
            names = list(range(len(data["events"])))[-ZONLY_EVENTS:]
            held = np.isin(np.asarray(data["train"]["events"]), names)
            held_out = {"kind": "events", "ids": names, "names": [data["events"][i] for i in names]}
            held_out["test_indices"] = np.flatnonzero(np.isin(np.asarray(data["test"]["events"]), names)).tolist()
            held_out["validation_indices"] = np.flatnonzero(np.isin(np.asarray(data["validation"]["events"]), names)).tolist()
        else:
            held = labels == ZONLY_CLASS
            held_out = {"kind": "class", "ids": [ZONLY_CLASS], "names": [data["classes"][ZONLY_CLASS]]}
            held_out["test_indices"] = np.flatnonzero(np.asarray(data["test"]["labels"]) == ZONLY_CLASS).tolist()
            held_out["validation_indices"] = np.flatnonzero(np.asarray(data["validation"]["labels"]) == ZONLY_CLASS).tolist()
    per_forward = int(held.sum()) // (len(forward) * planes)       # held-out examples per forward-only satellite
    usable = np.flatnonzero(~held)
    # Every satellite holds the same number of examples: as many as the data allow once the held-out examples
    # that can be dealt out evenly are counted in (446 tweets or 135 images in both variants of the real data).
    per_satellite = (len(usable) + per_forward * len(forward) * planes) // roles.count
    sizes = [per_satellite - (per_forward if roles.kind(slot) == "Z" else 0) for slot in range(per_plane)]
    if task.dataset == "humaid":
        chosen, summary, unused = humaid_planes(labels, np.asarray(data["train"]["events"], dtype=np.int64), usable, planes, sum(sizes), seed)
    else:
        chosen, summary, unused = eurosat_planes(labels, usable, classes, planes, sum(sizes), seed)
    held_order = np.flatnonzero(held)
    held_order = held_order[_rng(seed, "sat-partition", task.dataset, "held-out").permutation(len(held_order))]
    satellites, dealt = [], 0
    for plane in range(planes):
        parts, drawn = split_among_satellites(chosen[plane], labels, classes, sizes, (seed, "sat-partition", task.dataset, "satellite", plane))
        summary[plane]["label_counts"] = np.bincount(labels[chosen[plane]], minlength=classes).tolist()
        if task.dataset == "humaid":
            summary[plane]["event_counts"] = np.bincount(np.asarray(data["train"]["events"])[chosen[plane]], minlength=len(data["events"])).tolist()
        for slot in range(per_plane):
            sid, indices, extra = roles.sid(plane, slot), parts[slot], []
            if roles.kind(sid) == "Z" and per_forward:
                extra = held_order[dealt:dealt + per_forward].tolist()
                dealt += per_forward
            indices = indices + extra
            satellites.append({"sid": sid, "plane": plane, "slot": slot, "kind": roles.kind(sid), "indices": indices,
                               "data_seed": stable_seed(seed, "sat-data", sid), "held_out_examples": len(extra),
                               "label_counts": np.bincount(labels[indices], minlength=classes).tolist(),
                               "target_label_probabilities": drawn[slot]})
    unused = sorted(unused + held_order[dealt:].tolist())
    if held_out is not None:
        held_out.update(train_examples=int(held.sum()), per_forward_satellite=per_forward)
    flat = [index for record in satellites for index in record["indices"]]
    assert len(flat) == len(set(flat)) == roles.count * per_satellite and not set(flat) & set(unused)
    assert len({len(record["indices"]) for record in satellites}) == 1
    manifest = {"schema": 1, "task": task.key, "dataset": task.dataset, "variant": variant, "seed": seed, "planes": planes,
            "per_plane": per_plane, "examples_per_satellite": per_satellite, "train_examples": total,
            "classes": list(data["classes"]), "data_sha256": data["meta"]["sha256"], "labels_sha256": labels_hash(data),
            "parameters": {"main_share": MAIN_SHARE[0] / MAIN_SHARE[1], "plane_alpha": PLANE_ALPHA, "satellite_alpha": SATELLITE_ALPHA,
                           "seed_scheme": "numpy default_rng(stable_seed(seed, 'sat-partition', dataset, ...)); streams stable_seed(seed, 'sat-data', sid)"},
            "software": {"numpy": np.__version__}, "plane_summary": summary, "held_out": held_out,
            "unused_indices": unused, "satellites": satellites}
    manifest["partition_sha256"] = partition_hash(manifest)
    return manifest


def load_manifest(cfg, data):
    """The partition file of the configuration (written on first use), its sha256 and its raw bytes."""
    path = cfg.manifest_path
    if not path.exists():
        write_json(path, build_manifest(cfg.spec, data, cfg.seed, cfg.data_variant, cfg.constellation["planes"], cfg.constellation["per_plane"]))
    raw = path.read_bytes()
    manifest = json.loads(raw)
    count = cfg.satellites
    if manifest["labels_sha256"] != labels_hash(data):
        raise ValueError(f"{path} was built from other labels than the ones loaded now")
    if manifest["partition_sha256"] != partition_hash(manifest):
        raise ValueError(f"{path} does not match its recorded partition hash")
    if (manifest["task"], manifest["variant"], manifest["seed"]) != (cfg.spec.key, cfg.data_variant, cfg.seed):
        raise ValueError(f"{path} belongs to another task, variant or seed")
    assert [record["sid"] for record in manifest["satellites"]] == list(range(count))
    assert all(len(record["indices"]) >= cfg.batch_size for record in manifest["satellites"])
    return manifest, hashlib.sha256(raw).hexdigest(), raw


def manifest_statistics(manifest):
    """What the milestone report shows about a partition: sizes and how skewed planes and satellites are."""
    satellites, classes = manifest["satellites"], len(manifest["classes"])
    counts = np.array([record["label_counts"] for record in satellites], dtype=np.float64)
    planes = counts.reshape(manifest["planes"], manifest["per_plane"], classes).sum(1)
    overall = planes.sum(0) / planes.sum()

    def distance(rows, reference):
        """Mean total-variation distance between class distributions and a reference distribution."""
        shares = rows / rows.sum(-1, keepdims=True)
        return float(0.5 * np.abs(shares - reference).sum(-1).mean())

    plane_shares = planes / planes.sum(-1, keepdims=True)
    within = float(np.mean([0.5 * np.abs(counts[i] / counts[i].sum() - plane_shares[satellites[i]["plane"]]).sum() for i in range(len(satellites))]))
    stats = {"satellites": len(satellites), "examples_per_satellite": manifest["examples_per_satellite"],
             "used_examples": int(counts.sum()), "unused_examples": len(manifest["unused_indices"]),
             "class_share_overall": overall.tolist(), "plane_label_counts": planes.astype(int).tolist(),
             "tv_plane_to_overall": distance(planes, overall), "tv_satellite_to_plane": within,
             "classes_per_satellite_min": int((counts > 0).sum(1).min()), "classes_per_satellite_mean": float((counts > 0).sum(1).mean()),
             "share_of_data_by_class": {kind: float(sum(len(r["indices"]) for r in satellites if r["kind"] == kind) / counts.sum()) for kind in "BHZ"}}
    if manifest["dataset"] == "humaid":
        stats["main_share_by_plane"] = [round(entry["main_taken"] / (manifest["examples_per_satellite"] * manifest["per_plane"]), 4)
                                        for entry in manifest["plane_summary"]]
    if manifest["held_out"]:
        stats["held_out"] = {key: manifest["held_out"][key] for key in ("kind", "names", "train_examples", "per_forward_satellite")}
        stats["held_out"]["test_examples"] = len(manifest["held_out"]["test_indices"])
    return stats


def render_statistics(cfg, data, manifest, manifest_id):
    """The partition of a configuration as Markdown: what the milestone report quotes."""
    stats = manifest_statistics(manifest)
    lines = [f"Partition of task {cfg.task} ({cfg.spec.dataset}), variant {cfg.data_variant}, seed {cfg.seed}", "",
             f"- data: {data['meta']['source']} at {data['meta']['revision']}, content sha256 `{data['meta']['sha256']}`",
             f"- partition sha256 `{manifest['partition_sha256']}` (the examples of every satellite, nothing else); manifest file "
             f"`{cfg.manifest_path}`, sha256 `{manifest_id}`",
             f"- {stats['satellites']} satellites with {stats['examples_per_satellite']} training examples each; "
             f"{stats['used_examples']} used, {stats['unused_examples']} unused",
             "- share of the training data by satellite class: " + ", ".join(f"{kind} {share:.0%}" for kind, share in stats["share_of_data_by_class"].items()),
             f"- skew (mean total-variation distance of class distributions): plane to all data {stats['tv_plane_to_overall']:.3f}, "
             f"satellite to its plane {stats['tv_satellite_to_plane']:.3f}",
             f"- classes present on a satellite: at least {stats['classes_per_satellite_min']}, on average "
             f"{stats['classes_per_satellite_mean']:.1f} of {len(manifest['classes'])}"]
    if "held_out" in stats:
        held = stats["held_out"]
        lines.append(f"- held out for the forward-only satellites: {held['kind']} {', '.join(held['names'])}: {held['train_examples']} training "
                     f"examples, {held['per_forward_satellite']} per forward-only satellite, {held['test_examples']} test examples")
    lines += ["", "| plane | " + ("main events (taken / available) | " if cfg.spec.dataset == "humaid" else "")
              + " | ".join(str(c) for c in range(len(manifest["classes"]))) + " |",
              "| --- | " + ("--- | " if cfg.spec.dataset == "humaid" else "") + " | ".join("---" for _ in manifest["classes"]) + " |"]
    for entry in manifest["plane_summary"]:
        events = ""
        if cfg.spec.dataset == "humaid":
            events = ", ".join(data["events"][event] for event in entry["main_events"]) + f" ({entry['main_taken']} / {entry['main_available']}) | "
        lines.append(f"| {entry['plane']} | {events}" + " | ".join(str(count) for count in entry["label_counts"]) + " |")
    lines.append("\nClasses: " + ", ".join(f"{index} {name}" for index, name in enumerate(manifest["classes"])))
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Build (or read) the data cache and the partition of a configuration and print the partition statistics."""
    from .config import parse_sat_config

    cfg = parse_sat_config(argv)
    data = load_task_data(cfg)
    manifest, manifest_id, _ = load_manifest(cfg, data)
    print(render_statistics(cfg, data, manifest, manifest_id), flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
