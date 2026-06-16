"""
Build / refresh the static ablation viewer at /root/ablation_site/.

This builder is *manifest-only*: it never touches the HTML, CSS, or JS in
`index.html`. It locates the single line

    <script>const MANIFEST = {...};</script>

and replaces it with a freshly-computed manifest. Everything else in
`index.html` (layout, controls, card rendering, schedule SVG renderer, ...)
is left untouched, so iterative UI edits are preserved across rebuilds.

Source of truth:
    - /root/mof_multi_teacher.py   (PROMPTS, build_experiment_specs, ...)
    - /root/outputs_multi_teacher/ (the actual PNG files we already generated)

Outputs:
    /root/ablation_site/
        images/<all PNGs that exist on disk>
        index.html        (MANIFEST replaced; rest preserved)
        prompts.txt       (mirror of outputs_multi_teacher/prompts.txt)
        scripts/{mof_multi_teacher.py, build_ablation_site.py, requirements.txt}
        README.md         (created if missing)

Run: python /root/build_ablation_site.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from typing import Any, Optional

# Path layout is configured at runtime via _configure() (see parse_args()).
# Historically hardcoded to /root with outputs in /root/outputs_multi_teacher;
# now --site-dir / --src-dir / --mof-dir let it refresh a bundle anywhere (e.g.
# on shared storage). ``mof`` (the spec/prompt module) is imported in _configure
# after --mof-dir is on sys.path, so importing this file never needs torch.
_DEFAULT_SITE = "/apdcephfs_zwfy8/share_305110755/hunyuan/bowenping/jobs/ablation_site"

ROOT = "/root"            # dir holding mof_multi_teacher.py (added to sys.path)
SRC_DIR = ""              # dir holding the p{idx}__{tag}.png outputs
DST_DIR = ""             # site bundle to (re)build
DST_IMG_DIR = ""
DST_SCRIPTS_DIR = ""
INDEX_HTML = ""
mof: Any = None           # mof_multi_teacher module, imported in _configure()


def _configure(site_dir: str, src_dir: Optional[str], mof_dir: Optional[str]) -> None:
    """Resolve all path globals and import the mof_multi_teacher spec module.

    Defaults: ``src_dir`` -> ``<site>/images`` (refresh in place from the bundle's
    own PNGs) and ``mof_dir`` -> ``<site>/scripts`` (where the bundle keeps its copy
    of mof_multi_teacher.py).
    """
    global ROOT, SRC_DIR, DST_DIR, DST_IMG_DIR, DST_SCRIPTS_DIR, INDEX_HTML, mof
    import importlib

    DST_DIR = os.path.abspath(site_dir)
    DST_IMG_DIR = os.path.join(DST_DIR, "images")
    DST_SCRIPTS_DIR = os.path.join(DST_DIR, "scripts")
    INDEX_HTML = os.path.join(DST_DIR, "index.html")
    SRC_DIR = os.path.abspath(src_dir) if src_dir else DST_IMG_DIR
    ROOT = os.path.abspath(mof_dir) if mof_dir else DST_SCRIPTS_DIR
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    mof = importlib.import_module("mof_multi_teacher")


# ---------------------------------------------------------------------------
# Filename grammar (matches what mof_multi_teacher.main writes):
#   p{idx}__{tag}.png            -> seed slot 0
#   p{idx}__{tag}__seed{n}.png   -> seed slot n
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"^p(\d+)__(.+?)(?:__seed(\d+))?\.png$")

T_DEFAULT = 40           # mof_multi_teacher.main hard-codes T=40
SEED_SLOTS = [0, 1]      # ditto


# ---------------------------------------------------------------------------
# Tag -> rich metadata, used by the SPA to drive its dropdowns and labels.
# Order in this dict is the order we want optgroups to expose to the user.
# ---------------------------------------------------------------------------
def _tag_meta(tag: str) -> dict:
    """Heuristic mapping from tag string -> dict with strategy / teacher / etc.

    Mirrors what the previous `build_ablation_site.py` produced, plus a few
    extra strategies for the upcoming CFG-style configs (cfg_ovr, cfg_pair,
    cfg_tv, cfg_staged). Unknown tags fall back to strategy = "other".
    """
    if tag == "base_noLoRA":
        return {"strategy": "base", "teacher": None, "order": None,
                "window": None, "ratio": None, "strength": 0.0,
                "label": "base SD3.5"}
    if tag == "uniform":
        return {"strategy": "uniform", "teacher": None, "order": None,
                "window": None, "ratio": None, "strength": 0.33,
                "label": "uniform 1/K"}
    m = re.match(r"^single_(.+)$", tag)
    if m:
        return {"strategy": "single", "teacher": m.group(1), "order": None,
                "window": "always", "ratio": None, "strength": 1.0,
                "label": f"{m.group(1)} (single 1.0)"}
    m = re.match(r"^biased0\.8_(.+)$", tag)
    if m:
        return {"strategy": "biased", "teacher": m.group(1), "order": None,
                "window": None, "ratio": None, "strength": 0.8,
                "label": f"{m.group(1)} (biased 0.8)"}
    m = re.match(r"^staged_(.+)_(\d+(?:-\d+)+)$", tag)
    if m:
        order = m.group(1).split("-")
        return {"strategy": "staged3", "teacher": None,
                "order": order, "window": None,
                "ratio": m.group(2), "strength": None,
                "label": " \u2192 ".join(order) + f" ({m.group(2).replace('-', ':')})"}
    m = re.match(r"^(early|mid|late)Only_(.+)$", tag)
    if m:
        return {"strategy": "window", "teacher": m.group(2), "order": None,
                "window": m.group(1), "ratio": None, "strength": None,
                "label": f"{m.group(2)} ({m.group(1)} 1/3)"}
    m = re.match(r"^pair_(.+)_(\d+-\d+)$", tag)
    if m:
        order = m.group(1).split("-")
        return {"strategy": "pair", "teacher": None, "order": order,
                "window": None, "ratio": m.group(2), "strength": None,
                "label": f"{order[0]} \u2192 {order[1]} "
                         f"({m.group(2).replace('-', ':')})"}

    # CFG one-vs-rest constant.   e.g. cfgOvR_OCR_s1.5  or  cfgOvR_OCR_s-0.25
    m = re.match(r"^cfgOvR_(.+?)_s(-?[0-9]+(?:\.[0-9]+)?)$", tag)
    if m:
        teacher = m.group(1)
        s = float(m.group(2))
        return {"strategy": "cfg_ovr", "teacher": teacher, "order": None,
                "window": "always", "ratio": None, "strength": s,
                "label": f"{teacher} +/- (s={s:+.2f})"}

    # CFG one-vs-rest in a single window.   cfgOvR_OCR_s1.5_earlyOnly
    m = re.match(r"^cfgOvR_(.+?)_s(-?[0-9]+(?:\.[0-9]+)?)_(earlyOnly|midOnly|lateOnly)$", tag)
    if m:
        teacher = m.group(1); s = float(m.group(2)); win = m.group(3)
        return {"strategy": "cfg_tv", "teacher": teacher, "order": None,
                "window": win, "ratio": None, "strength": s,
                "label": f"{teacher} {win} (s={s:.2f})"}

    # CFG one-vs-rest with linear ramp.   cfgOvR_OCR_s_linup1.5
    m = re.match(r"^cfgOvR_(.+?)_s_(linup|lindown)(-?[0-9]+(?:\.[0-9]+)?)$", tag)
    if m:
        teacher = m.group(1); shape = m.group(2); s = float(m.group(3))
        return {"strategy": "cfg_tv", "teacher": teacher, "order": None,
                "window": shape, "ratio": None, "strength": s,
                "label": f"{teacher} {shape} 0\u21940\u2194{s:g}"}

    # CFG pairwise.       cfgPair_OCR-PickScore_s1.0
    m = re.match(r"^cfgPair_(.+?)-(.+?)_s(-?[0-9]+(?:\.[0-9]+)?)$", tag)
    if m:
        pos, neg, s = m.group(1), m.group(2), float(m.group(3))
        return {"strategy": "cfg_pair", "teacher": None, "order": [pos, neg],
                "window": None, "ratio": None, "strength": s,
                "label": f"+{pos} \u2212{neg} (s={s:.2f})"}

    # Staged CFG.   cfgStaged_GenEval-OCR-PickScore_s1.0  /  cfgStaged_layout-then-aesthetic
    m = re.match(r"^cfgStaged_(.+)$", tag)
    if m:
        body = m.group(1)
        # Try to peel off a trailing scale.
        ms = re.match(r"^(.+?)_s(-?[0-9]+(?:\.[0-9]+)?)$", body)
        if ms:
            label_body = ms.group(1).replace("-", " \u2192 ")
            s = float(ms.group(2))
            return {"strategy": "cfg_staged", "teacher": None,
                    "order": ms.group(1).split("-"),
                    "window": None, "ratio": None, "strength": s,
                    "label": f"{label_body} (s={s:.2f})"}
        return {"strategy": "cfg_staged", "teacher": None,
                "order": body.split("-"),
                "window": None, "ratio": None, "strength": None,
                "label": body.replace("-", " \u2192 ")}
    return {"strategy": "other", "teacher": None, "order": None,
            "window": None, "ratio": None, "strength": None,
            "label": tag}


# ---------------------------------------------------------------------------
# Convert a raw spec triple (tag, t_list, W) -> schedule snippet for the SPA.
# We always re-map column indices to the *global* teacher list
# [GenEval, OCR, PickScore] so the front end can use a single colour map.
# Slots whose adapter is None are dropped (zero contribution).
# ---------------------------------------------------------------------------
def _expand_W_to_global(t_list, W, short_names, teacher_names):
    """W: torch.Tensor (T, K'). Returns a 2D Python list (T, K_global) of floats."""
    import torch  # local import to avoid hard dep when unused

    T, Kp = W.shape
    K = len(short_names)

    # Build mapping local-col -> global-col (or None if "no adapter" slot).
    # `t_list[i]` is either the adapter name ("teacher1" / "teacher2" ...) or None.
    name_to_global = {tn: short_names.index(sn)
                      for tn, sn in zip(teacher_names, short_names)}
    map_to_global: list[Optional[int]] = []
    for slot in t_list:
        if slot is None:
            map_to_global.append(None)
        else:
            map_to_global.append(name_to_global[slot])

    Wg = [[0.0] * K for _ in range(T)]
    Wf = W.detach().to(torch.float32).cpu().tolist()
    for t in range(T):
        row = Wf[t]
        for col, g in enumerate(map_to_global):
            if g is None:
                continue
            Wg[t][g] += float(row[col])
    return Wg


# ---------------------------------------------------------------------------
# Index existing PNGs.
# ---------------------------------------------------------------------------
def collect_files(src_dir: str):
    """Return dict: tag -> {prompt_idx -> {seed -> filename}}."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    if not os.path.isdir(src_dir):
        return out
    for fn in sorted(os.listdir(src_dir)):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        p = m.group(1)
        tag = m.group(2)
        seed = m.group(3) if m.group(3) is not None else "0"
        out.setdefault(tag, {}).setdefault(p, {})[seed] = fn
    return out


def load_prompts(path: str) -> dict[str, str]:
    res: dict[str, str] = {}
    if not os.path.exists(path):
        return res
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, val = line.partition("\t")
            if key.startswith("p") and key[1:].isdigit():
                res[key[1:]] = val
    return res


# ---------------------------------------------------------------------------
# Comparison axes (same as before, with extra CFG axes appended).
# Templates use `{T}` (current teacher), `{A}/{B}` (current pair members).
# ---------------------------------------------------------------------------
def build_comparisons(tag_meta: "OrderedDict[str, dict]",
                      teachers: list[str]) -> list[dict]:
    has = lambda t: t in tag_meta  # noqa: E731
    cmps: list[dict] = []

    if has("single_GenEval"):
        cmps.append({
            "id": "teacher_single", "title": "Teacher (single, full strength)",
            "group": "teacher", "varyKind": "tag", "varyLabel": "teacher",
            "desc": "One teacher active full-time. Which teacher helps this prompt?",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": t, "tag": f"single_{t}"} for t in teachers
                        if has(f"single_{t}")],
        })
    if has("biased0.8_GenEval"):
        cmps.append({
            "id": "teacher_biased", "title": "Teacher (biased 0.8)",
            "group": "teacher", "varyKind": "tag", "varyLabel": "teacher",
            "desc": "One teacher at 0.8, the rest at 0.1 each. Vary which is dominant.",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": t, "tag": f"biased0.8_{t}"} for t in teachers
                        if has(f"biased0.8_{t}")],
        })
        cmps.append({
            "id": "strength", "title": "Strength / dominance of one teacher",
            "group": "teacher", "varyKind": "tag", "varyLabel": "strength",
            "desc": "Hold the teacher fixed; raise its dominance 0 \u2192 0.8 \u2192 1.0.",
            "holds": ["teacher"],
            "refs": [{"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": "0.0 (base)", "tag": "base_noLoRA"},
                        {"label": "0.8 (biased)", "tag": "biased0.8_{T}"},
                        {"label": "1.0 (single)", "tag": "single_{T}"}],
        })
    if has("earlyOnly_GenEval"):
        cmps.append({
            "id": "window", "title": "Injection window for one teacher",
            "group": "teacher", "varyKind": "tag", "varyLabel": "denoising window",
            "desc": "Hold the teacher fixed; vary WHEN it is active along denoising.",
            "holds": ["teacher"],
            "refs": [{"label": "base", "tag": "base_noLoRA"}],
            "columns": [{"label": "early 1/3", "tag": "earlyOnly_{T}"},
                        {"label": "mid 1/3",   "tag": "midOnly_{T}"},
                        {"label": "late 1/3",  "tag": "lateOnly_{T}"},
                        {"label": "always",    "tag": "single_{T}"}],
        })
    if has("staged_GenEval-OCR-PickScore_1-1-1"):
        # Six permutations: build dynamically from teachers.
        import itertools
        ord_cols = []
        for perm in itertools.permutations(teachers):
            tg = "staged_" + "-".join(perm) + "_1-1-1"
            if has(tg):
                ord_cols.append({"label": " \u2192 ".join(perm), "tag": tg})
        cmps.append({
            "id": "staged_order", "title": "Stage ordering (equal 1:1:1 thirds)",
            "group": "schedule", "varyKind": "tag", "varyLabel": "ordering",
            "desc": "Three equal stages, one teacher each. Vary the hand-off order.",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": ord_cols,
        })
    if has("staged_GenEval-OCR-PickScore_1-1-2"):
        cmps.append({
            "id": "stage_ratio", "title": "Stage ratio (default order)",
            "group": "schedule", "varyKind": "tag", "varyLabel": "stage ratio",
            "desc": "Order fixed (GenEval \u2192 OCR \u2192 PickScore); vary stage lengths.",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": "1:1:1",
                         "tag": "staged_GenEval-OCR-PickScore_1-1-1"},
                        {"label": "1:1:2",
                         "tag": "staged_GenEval-OCR-PickScore_1-1-2"}],
        })
    # pair_which: one tag per unordered pair, 1-1 split.
    pair_cols = []
    import itertools
    for a, b in itertools.combinations(teachers, 2):
        tg = f"pair_{a}-{b}_1-1"
        if has(tg):
            pair_cols.append({"label": f"{a} \u2192 {b}", "tag": tg})
    if pair_cols:
        cmps.append({
            "id": "pair_which",
            "title": "Which two-teacher pair (equal halves)",
            "group": "schedule", "varyKind": "tag", "varyLabel": "pair",
            "desc": "Two teachers in equal halves; the third is excluded. Vary the pair.",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"}],
            "columns": pair_cols,
        })
    if has("pair_GenEval-OCR_1-1") and has("pair_OCR-GenEval_1-2"):
        cmps.append({
            "id": "pair_ratio", "title": "Pair split / ordering",
            "group": "schedule", "varyKind": "tag", "varyLabel": "split / order",
            "desc": "Hold the pair fixed; vary the stage split and the ordering.",
            "holds": ["pair"],
            "refs": [{"label": "base", "tag": "base_noLoRA"}],
            "columns": [{"label": "A \u2192 B (1:1)", "tag": "pair_{A}-{B}_1-1"},
                        {"label": "A \u2192 B (1:2)", "tag": "pair_{A}-{B}_1-2"},
                        {"label": "B \u2192 A (1:2)", "tag": "pair_{B}-{A}_1-2"}],
        })
    if has("earlyOnly_GenEval"):
        cmps.append({
            "id": "teacher_x_window",
            "title": "Teacher \u00d7 window (2-variable grid)",
            "group": "schedule", "varyKind": "grid",
            "varyLabel": "teacher \u00d7 window",
            "desc": "Rows = teacher, columns = early / mid / late single-teacher injection.",
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"}],
            "columns": [],
        })

    # ---- New CFG axes (appear once at least one CFG tag exists) -----------
    cfg_ovr_tags = sorted(t for t in tag_meta
                          if tag_meta[t]["strategy"] == "cfg_ovr")
    if cfg_ovr_tags:
        # group by teacher; sweep strength.
        from collections import defaultdict
        per_t: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for tg in cfg_ovr_tags:
            meta = tag_meta[tg]
            per_t[meta["teacher"]].append((meta["strength"], tg))
        for t in teachers:
            if t not in per_t:
                continue
            per_t[t].sort(key=lambda x: x[0])
            cmps.append({
                "id": f"cfg_ovr_{t}",
                "title": f"CFG one-vs-rest strength sweep ({t})",
                "group": "cfg", "varyKind": "tag",
                "varyLabel": "scale",
                "desc": (f"+{t} positive, all other teachers split equal "
                         f"negative weight; rows still sum to 1. "
                         f"Sweep the strength."),
                "holds": [],
                "refs": [{"label": "base", "tag": "base_noLoRA"},
                         {"label": "uniform", "tag": "uniform"},
                         {"label": "single 1.0", "tag": f"single_{t}"}],
                "columns": [{"label": f"s = {s:+g}", "tag": tg}
                            for s, tg in per_t[t]],
            })
    cfg_pair_tags = sorted(t for t in tag_meta
                           if tag_meta[t]["strategy"] == "cfg_pair")
    if cfg_pair_tags:
        cmps.append({
            "id": "cfg_pair", "title": "CFG pairwise (one teacher vs another)",
            "group": "cfg", "varyKind": "tag", "varyLabel": "pair / strength",
            "desc": ("Pull towards one teacher and away from another (the "
                     "third gets weight 0). Rows sum to 1."),
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": tag_meta[tg]["label"], "tag": tg}
                        for tg in cfg_pair_tags],
        })
    cfg_tv_tags = sorted(t for t in tag_meta
                         if tag_meta[t]["strategy"] == "cfg_tv")
    if cfg_tv_tags:
        cmps.append({
            "id": "cfg_tv",
            "title": "Time-varying CFG (one-vs-rest in one window)",
            "group": "cfg", "varyKind": "tag",
            "varyLabel": "teacher \u00d7 window \u00d7 scale",
            "desc": ("CFG one-vs-rest applied only inside one denoising "
                     "window; the rest of the trajectory is base SD3.5."),
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"}],
            "columns": [{"label": tag_meta[tg]["label"], "tag": tg}
                        for tg in cfg_tv_tags],
        })
    cfg_staged_tags = sorted(t for t in tag_meta
                             if tag_meta[t]["strategy"] == "cfg_staged")
    if cfg_staged_tags:
        cmps.append({
            "id": "cfg_staged",
            "title": "Staged CFG (per-stage one-vs-rest)",
            "group": "cfg", "varyKind": "tag", "varyLabel": "ordering / scale",
            "desc": ("Three equal stages; each stage runs CFG one-vs-rest with "
                     "one positive teacher and two negative."),
            "holds": [],
            "refs": [{"label": "base", "tag": "base_noLoRA"},
                     {"label": "uniform", "tag": "uniform"},
                     {"label": "staged 1:1:1",
                      "tag": "staged_GenEval-OCR-PickScore_1-1-1"}],
            "columns": [{"label": tag_meta[tg]["label"], "tag": tg}
                        for tg in cfg_staged_tags],
        })

    # ---- Robustness axes -------------------------------------------------
    cmps.append({
        "id": "seed", "title": "Seed robustness",
        "group": "robustness", "varyKind": "seed", "varyLabel": "seed",
        "desc": "Hold one config + prompt fixed; compare the two seeds.",
        "holds": ["config"], "refs": [], "columns": [],
    })
    cmps.append({
        "id": "prompt", "title": "Across prompts",
        "group": "robustness", "varyKind": "prompt", "varyLabel": "prompt",
        "desc": "Hold one config + seed fixed; sweep every prompt.",
        "holds": ["config"], "refs": [], "columns": [],
    })
    return cmps


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------
def build_manifest() -> dict:
    short_names: list[str] = list(mof.TEACHER_SHORT_NAMES)
    teacher_names: list[str] = list(mof.TEACHER_ADAPTERS.keys())
    T = T_DEFAULT

    # 1) Get the canonical experiment specs (this is THE source of truth).
    specs = mof.build_experiment_specs(short_names, teacher_names, T)

    # 2) Build per-tag metadata + schedule (T, K_global) float matrix.
    tag_meta: "OrderedDict[str, dict]" = OrderedDict()
    for tag, t_list, W in specs:
        meta = _tag_meta(tag)
        meta["schedule"] = {
            "T": int(T),
            "teachers": list(short_names),
            "W": _expand_W_to_global(t_list, W, short_names, teacher_names),
        }
        tag_meta[tag] = meta

    # 3) Index PNG files (only those that actually exist).
    file_index = collect_files(SRC_DIR)
    # Drop tags that have no files at all? -> Keep them; the SPA shows "(n/a)".
    # But we still want every spec listed in `tags` so the UI can render the
    # schedule column even if the PNG is missing.

    # Keep only files whose tag is in our spec list (avoid stale tags).
    file_index = {tag: by_p for tag, by_p in file_index.items()
                  if tag in tag_meta}

    # 4) Prompts.
    prompts = load_prompts(os.path.join(SRC_DIR, "prompts.txt"))
    # Fallback to the in-source PROMPTS list if outputs/prompts.txt missing.
    if not prompts:
        prompts = {str(i): p for i, p in enumerate(mof.PROMPTS)}

    # 5) Pairs (every unordered pair of teachers).
    import itertools
    pairs = [list(p) for p in itertools.combinations(short_names, 2)]

    # 6) Comparisons.
    comparisons = build_comparisons(tag_meta, short_names)

    return {
        "teachers": short_names,
        "pairs": pairs,
        "seeds": SEED_SLOTS,
        "prompts": prompts,
        "tags": tag_meta,
        "index": file_index,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Asset copying & HTML in-place patching.
# ---------------------------------------------------------------------------
def copy_images(file_index: dict):
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    n_copied = 0
    for tag, by_p in file_index.items():
        for p, by_s in by_p.items():
            for seed, fn in by_s.items():
                src = os.path.join(SRC_DIR, fn)
                dst = os.path.join(DST_IMG_DIR, fn)
                if not os.path.exists(src):
                    continue
                if (not os.path.exists(dst)
                        or os.path.getmtime(src) > os.path.getmtime(dst)):
                    shutil.copy2(src, dst)
                    n_copied += 1
    return n_copied


def copy_scripts():
    os.makedirs(DST_SCRIPTS_DIR, exist_ok=True)
    for src_name in ("mof_multi_teacher.py", "build_ablation_site.py"):
        src = os.path.join(ROOT, src_name)
        dst = os.path.join(DST_SCRIPTS_DIR, src_name)
        # When refreshing in place, ROOT == <site>/scripts, so src and dst are the
        # same file; skip the self-copy (shutil raises SameFileError otherwise).
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
    # Pin a minimal requirements.txt for reproducibility.
    req = os.path.join(DST_SCRIPTS_DIR, "requirements.txt")
    if not os.path.exists(req):
        with open(req, "w") as f:
            f.write(
                "# Multi-teacher LoRA ablation viewer\n"
                "# (loose pins -- tested with the listed versions on the dev box)\n"
                "torch>=2.4\ndiffusers>=0.31\npeft>=0.13\ntransformers>=4.45\n"
                "accelerate>=0.34\nnumpy\nPillow\n"
            )


MANIFEST_LINE_RE = re.compile(
    r"<script>\s*const\s+MANIFEST\s*=\s*.*?;\s*</script>",
    re.DOTALL,
)


def patch_index_html(manifest: dict):
    if not os.path.exists(INDEX_HTML):
        raise SystemExit(
            f"{INDEX_HTML} not found. The viewer template must be present "
            f"before running this builder; this script only patches the "
            f"MANIFEST line, never rebuilds the layout."
        )
    with open(INDEX_HTML, "r") as f:
        html = f.read()

    payload = json.dumps(manifest, ensure_ascii=False, separators=(", ", ": "))
    new_line = "<script>const MANIFEST = " + payload + ";</script>"

    new_html, n = MANIFEST_LINE_RE.subn(new_line, html, count=1)
    if n == 0:
        raise SystemExit(
            "Could not locate the `<script>const MANIFEST = ...;</script>` "
            "line in index.html. Please make sure that exact marker is present."
        )
    if new_html != html:
        with open(INDEX_HTML, "w") as f:
            f.write(new_html)


def write_prompts_txt(prompts: dict[str, str]):
    out = os.path.join(DST_DIR, "prompts.txt")
    with open(out, "w") as f:
        for k in sorted(prompts.keys(), key=int):
            f.write(f"p{k}\t{prompts[k]}\n")


def ensure_readme():
    out = os.path.join(DST_DIR, "README.md")
    if os.path.exists(out):
        return
    with open(out, "w") as f:
        f.write(
            "# Multi-teacher ablation viewer\n\n"
            "Open `index.html` in any browser. All assets are local.\n\n"
            "## Reproduce\n\n"
            "```\npip install -r scripts/requirements.txt\n"
            "python scripts/mof_multi_teacher.py\n"
            "python scripts/build_ablation_site.py\n```\n"
        )


def serve_site(site_dir: str, port: int, bind: str = "0.0.0.0") -> None:
    """Serve the ablation bundle over HTTP for port-forwarding (e.g. Cursor Ports).

    index.html references images by relative path, so the web root must be the
    bundle dir. Threaded so a comparison strip's images load concurrently.
    """
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    index = os.path.join(site_dir, "index.html")
    if not os.path.isfile(index):
        raise SystemExit(f"no index.html at {index}; build the site first")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=site_dir)
    httpd = ThreadingHTTPServer((bind, port), handler)
    httpd.daemon_threads = True
    print(
        f"[serve] {site_dir} at http://{bind}:{port}/index.html  "
        f"(forward port {port} in Cursor, open http://localhost:{port}/index.html; Ctrl-C to stop)"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[serve] stopped.")
    finally:
        httpd.server_close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build / refresh the multi-teacher ablation viewer (manifest-only) and serve it.",
    )
    p.add_argument(
        "--site-dir", default=_DEFAULT_SITE,
        help=f"Ablation bundle to (re)build (holds index.html + images/). Default: {_DEFAULT_SITE}",
    )
    p.add_argument(
        "--src-dir", default=None,
        help="Dir with the p{idx}__{tag}.png outputs. Default: <site-dir>/images (refresh in place).",
    )
    p.add_argument(
        "--mof-dir", default=None,
        help="Dir containing mof_multi_teacher.py. Default: <site-dir>/scripts.",
    )
    p.add_argument("--no-build", action="store_true", help="Skip rebuild; only --serve the bundle.")
    p.add_argument("--serve", action="store_true", help="Serve --site-dir after building. Blocks.")
    p.add_argument("--port", type=int, default=8000, help="Port for --serve (default 8000).")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address for --serve (default 0.0.0.0).")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_build:
        _configure(args.site_dir, args.src_dir, args.mof_dir)
        os.makedirs(DST_DIR, exist_ok=True)

        manifest = build_manifest()

        # 1. Sync image files (no-op when src == <site>/images).
        n_copied = copy_images(manifest["index"])
        print(f"[bundle] copied {n_copied} images "
              f"(of {sum(len(by_s) for by_p in manifest['index'].values() for by_s in by_p.values())} indexed)")

        # 2. Mirror prompts.txt + scripts/.
        write_prompts_txt(manifest["prompts"])
        copy_scripts()
        ensure_readme()

        # 3. Patch the MANIFEST line in-place; leave the rest of index.html alone.
        patch_index_html(manifest)
        print(f"[done ] patched MANIFEST in {INDEX_HTML}")
        print(f"        {len(manifest['tags'])} tags, "
              f"{len(manifest['comparisons'])} comparison axes, "
              f"{len(manifest['prompts'])} prompts")

    if args.serve:
        serve_site(os.path.abspath(args.site_dir), args.port, args.bind)


if __name__ == "__main__":
    main()
