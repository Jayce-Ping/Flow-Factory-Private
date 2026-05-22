#!/usr/bin/env python3
"""Fetch Flow-Factory-OPD-Experiments W&B metrics and emit docs/experiments.tex."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import wandb

DEFAULT_PROJECT = "Flow-Factory-OPD-Experiments"

DEFAULT_RUNS: Tuple[str, ...] = (
    "0_sd35-baseline",
    "1_geneval-baseline",
    "1_ocr-baseline",
    "1_pickscore-baseline",
    "2_geneval-ocr-baseline",
    "2_geneval-pickscore-baseline",
    "2_ocr-pickscore-baseline",
    "3_geneval-ocr-pickscore-pcgrad_residual",
)

MAIN_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("geneval_ge", "eval/geneval/reward_geneval_mean", "GenEval"),
    ("geneval_ps", "eval/geneval/reward_pick_score_mean", "PickScore"),
    ("pickscore_ps", "eval/pickscore/reward_pick_score_mean", "PickScore"),
    ("ocr_ocr", "eval/ocr/reward_ocr_mean", "OCR"),
    ("ocr_ps", "eval/ocr/reward_pick_score_mean", "PickScore"),
)

REQUIRED_WANDB_KEYS: Tuple[str, ...] = tuple(col[1] for col in MAIN_COLUMNS)

TEACHER_IN_DOMAIN_COLUMN: Dict[str, str] = {
    "geneval": "geneval_ge",
    "ocr": "ocr_ocr",
    "pickscore": "pickscore_ps",
}

SINGLE_TEACHER_BASELINE_RUNS: Dict[str, str] = {
    "geneval": "1_geneval-baseline",
    "ocr": "1_ocr-baseline",
    "pickscore": "1_pickscore-baseline",
}

DELTA_EPSILON = 1e-4

# (test_set, reward) pairs grayed in the appendix table per teacher
TEACHER_IN_DOMAIN_APPENDIX: Dict[str, Tuple[str, str]] = {
    "geneval": ("geneval", "geneval"),
    "ocr": ("ocr", "ocr"),
    "pickscore": ("pickscore", "pick_score"),
}

IN_DOMAIN_CELL_COLOR = "gray!20"

GENEVAL_TAGS: Tuple[str, ...] = (
    "color_attr",
    "colors",
    "counting",
    "position",
    "single_object",
    "two_object",
)

REWARD_WANDB_SEGMENT: Dict[str, str] = {
    "geneval": "geneval",
    "pick_score": "pick_score",
    "ocr": "ocr",
}

REWARD_DISPLAY: Dict[str, str] = {
    "geneval": "GenEval",
    "pick_score": "PickScore",
    "ocr": "OCR",
}

BENCHMARK_DISPLAY: Dict[str, str] = {
    "geneval": "GenEval benchmark",
    "pickscore": "PickScore benchmark",
    "ocr": "OCR benchmark",
}

TAG_DISPLAY: Dict[str, str] = {
    "color_attr": "Color attr",
    "colors": "Colors",
    "counting": "Counting",
    "position": "Position",
    "single_object": "Single obj",
    "two_object": "Two obj",
}

# (test_set, ((reward_key, (metric, ...)), ...))
APPENDIX_TABLE_STRUCTURE: Tuple[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]], ...] = (
    (
        "geneval",
        (
            ("geneval", ("std", *GENEVAL_TAGS)),
            ("pick_score", ("std", *GENEVAL_TAGS)),
        ),
    ),
    ("pickscore", (("pick_score", ("std",)),)),
    (
        "ocr",
        (
            ("ocr", ("std",)),
            ("pick_score", ("std",)),
        ),
    ),
)


@dataclass(frozen=True)
class AppendixColumn:
    wandb_key: str
    test_set: str
    reward: str
    metric: str


@dataclass(frozen=True)
class ParsedRun:
    name: str
    num_checkpoints: int
    teachers: Tuple[str, ...]
    is_pcgrad: bool

    @property
    def teachers_display(self) -> str:
        if self.num_checkpoints == 0:
            return "SD3.5 (base)"
        return "+".join(self.teachers)


def parse_run_name(name: str) -> ParsedRun:
    match = re.fullmatch(r"(\d+)_(.+)", name)
    if match is None:
        raise ValueError(
            f"expected run name matching '{{n}}_{{teachers}}-{{suffix}}', got {name!r}"
        )
    num_checkpoints = int(match.group(1))
    rest = match.group(2)

    if num_checkpoints == 0:
        if rest != "sd35-baseline":
            raise ValueError(f"expected 0_sd35-baseline for n=0 ensemble eval run, got {name!r}")
        return ParsedRun(
            name=name,
            num_checkpoints=0,
            teachers=(),
            is_pcgrad=False,
        )

    if rest.endswith("-pcgrad_residual"):
        is_pcgrad = True
        teacher_part = rest[: -len("-pcgrad_residual")]
    elif rest.endswith("-baseline"):
        is_pcgrad = False
        teacher_part = rest[: -len("-baseline")]
    else:
        raise ValueError(f"expected run suffix -baseline or -pcgrad_residual, got {name!r}")

    teachers = tuple(teacher_part.split("-")) if teacher_part else ()
    if len(teachers) != num_checkpoints:
        raise ValueError(
            f"run {name!r}: num_checkpoints={num_checkpoints} but parsed "
            f"teachers={teachers!r} (len={len(teachers)})"
        )
    return ParsedRun(
        name=name,
        num_checkpoints=num_checkpoints,
        teachers=teachers,
        is_pcgrad=is_pcgrad,
    )


def aggregation_label(parsed: ParsedRun) -> str:
    if parsed.num_checkpoints < 2:
        return r"---"
    if parsed.is_pcgrad:
        return r"\textsc{PCGrad}"
    return r"\textsc{Avg}"


def in_domain_column_ids(parsed: ParsedRun) -> frozenset[str]:
    return frozenset(
        TEACHER_IN_DOMAIN_COLUMN[teacher]
        for teacher in parsed.teachers
        if teacher in TEACHER_IN_DOMAIN_COLUMN
    )


def highlight_in_domain(parsed: ParsedRun, column_id: str) -> bool:
    return column_id in in_domain_column_ids(parsed)


def highlight_appendix_column(parsed: ParsedRun, column: AppendixColumn) -> bool:
    for teacher in parsed.teachers:
        in_domain = TEACHER_IN_DOMAIN_APPENDIX.get(teacher)
        if in_domain is None:
            continue
        test_set, reward = in_domain
        if column.test_set == test_set and column.reward == reward:
            return True
    return False


def format_metric(value: float, gray_background: bool = False) -> str:
    text = f"{value:.3f}"
    if gray_background:
        return rf"\cellcolor{{{IN_DOMAIN_CELL_COLOR}}}{text}"
    return text


def build_single_teacher_baselines(
    metrics_by_run: Mapping[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    baselines: Dict[str, Dict[str, float]] = {}
    for teacher, run_name in SINGLE_TEACHER_BASELINE_RUNS.items():
        if run_name not in metrics_by_run:
            raise ValueError(
                f"single-teacher baseline run {run_name!r} for teacher {teacher!r} "
                f"not found in fetched metrics"
            )
        run_metrics = metrics_by_run[run_name]
        teacher_baselines: Dict[str, float] = {}
        for col_id, wandb_key, _ in MAIN_COLUMNS:
            if wandb_key not in run_metrics:
                raise ValueError(
                    f"baseline run {run_name!r} missing metric {wandb_key!r} "
                    f"for main table column {col_id!r}"
                )
            teacher_baselines[col_id] = run_metrics[wandb_key]
        baselines[teacher] = teacher_baselines
    return baselines


def combo_row_baseline_value(
    parsed: ParsedRun,
    col_id: str,
    single_baselines: Mapping[str, Dict[str, float]],
) -> float:
    """Max metric among single-teacher baselines for teachers in this ensemble row."""
    if parsed.num_checkpoints < 2:
        raise ValueError(
            f"combo_row_baseline_value requires n>=2 teachers, got run {parsed.name!r}"
        )
    values = [
        single_baselines[teacher][col_id]
        for teacher in parsed.teachers
        if teacher in single_baselines
    ]
    if len(values) != len(parsed.teachers):
        missing = set(parsed.teachers) - set(single_baselines.keys())
        raise ValueError(
            f"run {parsed.name!r}: no single-teacher baseline for teachers {sorted(missing)}"
        )
    return max(values)


def format_delta_suffix(delta: float) -> str:
    if abs(delta) < DELTA_EPSILON:
        return ""
    if delta > 0:
        return rf"{{\footnotesize\textcolor{{blue}}{{{delta:+.3f}}}}}"
    return rf"{{\footnotesize\textcolor{{red}}{{{delta:.3f}}}}}"


def format_main_table_cell(
    value: float,
    parsed: ParsedRun,
    col_id: str,
    single_baselines: Mapping[str, Dict[str, float]],
) -> str:
    text = f"{value:.3f}"

    if parsed.num_checkpoints == 1:
        if highlight_in_domain(parsed, col_id):
            return f"\\textbf{{{text}}}"
        return text

    if parsed.num_checkpoints >= 2:
        baseline_value = combo_row_baseline_value(parsed, col_id, single_baselines)
        delta = value - baseline_value
        body = text + format_delta_suffix(delta)
        if highlight_in_domain(parsed, col_id):
            return rf"\cellcolor{{{IN_DOMAIN_CELL_COLOR}}}{body}"
        return body

    return text


def fetch_run_metrics(
    api: wandb.Api,
    project: str,
    run_name: str,
) -> Dict[str, float]:
    runs = [r for r in api.runs(project) if r.name == run_name]
    if not runs:
        raise ValueError(f"no W&B run named {run_name!r} in project {project!r}")
    if len(runs) > 1:
        raise ValueError(
            f"multiple W&B runs named {run_name!r} in project {project!r}: "
            f"{[r.id for r in runs]}"
        )
    run = runs[0]
    history = run.history(samples=1000)
    if history.empty:
        raise ValueError(f"run {run_name!r} ({run.id}) has empty history")

    metrics: Dict[str, float] = {}
    for col in history.columns:
        if not col.startswith("eval/"):
            continue
        if col.endswith("/eval_samples"):
            continue
        series = history[col].dropna()
        if series.empty:
            continue
        value = float(series.iloc[-1])
        metrics[col] = value

    missing = [k for k in REQUIRED_WANDB_KEYS if k not in metrics]
    if missing:
        raise ValueError(f"run {run_name!r} missing required eval metrics: {missing}")
    return metrics


def wandb_key_for_appendix_metric(test_set: str, reward: str, metric: str) -> str:
    segment = REWARD_WANDB_SEGMENT[reward]
    if metric == "std":
        return f"eval/{test_set}/reward_{segment}_std"
    return f"eval/{test_set}/reward_{segment}/{metric}_mean"


def build_appendix_columns(all_keys: Sequence[str]) -> List[AppendixColumn]:
    key_set = set(all_keys)
    columns: List[AppendixColumn] = []
    for test_set, reward_groups in APPENDIX_TABLE_STRUCTURE:
        for reward, metrics in reward_groups:
            for metric in metrics:
                wandb_key = wandb_key_for_appendix_metric(test_set, reward, metric)
                if wandb_key not in key_set:
                    raise ValueError(
                        f"expected W&B metric {wandb_key!r} for appendix table "
                        f"(test_set={test_set!r}, reward={reward!r}, metric={metric!r}) "
                        f"but it was not found in run history"
                    )
                columns.append(
                    AppendixColumn(
                        wandb_key=wandb_key,
                        test_set=test_set,
                        reward=reward,
                        metric=metric,
                    )
                )
    return columns


def metric_header_label(metric: str) -> str:
    if metric == "std":
        return "Std"
    return TAG_DISPLAY.get(metric, metric.replace("_", " "))


def _group_span_end(
    columns: Sequence[AppendixColumn],
    start: int,
    *,
    test_set: Optional[str] = None,
    reward: Optional[str] = None,
) -> int:
    end = start
    while end < len(columns):
        col = columns[end]
        if test_set is not None and col.test_set != test_set:
            break
        if reward is not None and col.reward != reward:
            break
        end += 1
    return end


def _cmidrule(start_col: int, end_col: int) -> str:
    return rf"\cmidrule(lr){{{start_col}-{end_col}}}"


def build_appendix_header_rows(
    columns: Sequence[AppendixColumn],
    *,
    setup_cols: int = 2,
) -> Tuple[str, str, str, str, str]:
    """Return (row1, row2, row3, benchmark_cmidrules, reward_cmidrules) LaTeX lines."""
    first_data_col = setup_cols + 1

    row1 = [r"\multirow{3}{*}{Teachers}", r"\multirow{3}{*}{Agg.}"]
    row2 = ["", ""]
    row3 = ["", ""]

    benchmark_rules: List[str] = []
    reward_rules: List[str] = []

    idx = 0
    while idx < len(columns):
        test_set = columns[idx].test_set
        bench_end = _group_span_end(columns, idx, test_set=test_set)
        bench_span = bench_end - idx
        row1.append(rf"\multicolumn{{{bench_span}}}{{c}}{{{BENCHMARK_DISPLAY[test_set]}}}")
        bench_start_col = first_data_col + idx
        bench_end_col = first_data_col + bench_end - 1
        benchmark_rules.append(_cmidrule(bench_start_col, bench_end_col))

        r_idx = idx
        while r_idx < bench_end:
            reward = columns[r_idx].reward
            reward_end = _group_span_end(columns, r_idx, test_set=test_set, reward=reward)
            reward_span = reward_end - r_idx
            row2.append(rf"\multicolumn{{{reward_span}}}{{c}}{{{REWARD_DISPLAY[reward]}}}")
            reward_start_col = first_data_col + r_idx
            reward_end_col = first_data_col + reward_end - 1
            reward_rules.append(_cmidrule(reward_start_col, reward_end_col))

            for col in columns[r_idx:reward_end]:
                row3.append(metric_header_label(col.metric))
            r_idx = reward_end
        idx = bench_end

    row1_line = " & ".join(row1) + r" \\"
    row2_line = " & ".join(row2) + r" \\"
    row3_line = " & ".join(row3) + r" \\"
    return (
        row1_line,
        row2_line,
        row3_line,
        " ".join(benchmark_rules),
        " ".join(reward_rules),
    )


def build_main_table_rows(
    runs: Sequence[str],
    metrics_by_run: Mapping[str, Dict[str, float]],
    single_baselines: Mapping[str, Dict[str, float]],
) -> List[str]:
    rows: List[str] = []
    for run_name in runs:
        parsed = parse_run_name(run_name)
        metrics = metrics_by_run[run_name]
        cells = [
            parsed.teachers_display.replace("_", r"\_"),
            aggregation_label(parsed),
        ]
        for col_id, wandb_key, _ in MAIN_COLUMNS:
            value = metrics[wandb_key]
            cells.append(
                format_main_table_cell(
                    value,
                    parsed,
                    col_id,
                    single_baselines,
                )
            )
        rows.append(" & ".join(cells) + r" \\")
    return rows


def build_appendix_table_rows(
    runs: Sequence[str],
    metrics_by_run: Mapping[str, Dict[str, float]],
    appendix_columns: Sequence[AppendixColumn],
) -> List[str]:
    rows: List[str] = []
    for run_name in runs:
        parsed = parse_run_name(run_name)
        metrics = metrics_by_run[run_name]
        cells = [
            parsed.teachers_display.replace("_", r"\_"),
            aggregation_label(parsed),
        ]
        for column in appendix_columns:
            key = column.wandb_key
            if key not in metrics:
                cells.append("---")
                continue
            cells.append(
                format_metric(
                    metrics[key],
                    gray_background=highlight_appendix_column(parsed, column),
                )
            )
        rows.append(" & ".join(cells) + r" \\")
    return rows


def generate_experiments_tex(
    *,
    project: str,
    runs: Sequence[str],
    metrics_by_run: Mapping[str, Dict[str, float]],
    fetched_at: str,
) -> str:
    all_keys = sorted({k for m in metrics_by_run.values() for k in m})
    appendix_columns = build_appendix_columns(all_keys)
    single_baselines = build_single_teacher_baselines(metrics_by_run)
    main_rows = build_main_table_rows(runs, metrics_by_run, single_baselines)
    appendix_rows = build_appendix_table_rows(runs, metrics_by_run, appendix_columns)
    (
        appendix_row1,
        appendix_row2,
        appendix_row3,
        appendix_bench_cmid,
        appendix_reward_cmid,
    ) = build_appendix_header_rows(appendix_columns)

    appendix_col_spec = "ll" + "c" * len(appendix_columns)

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\usepackage[table]{xcolor}",
        "",
        r"\title{Ensemble Eval Experiments}",
        r"\author{Flow-Factory}",
        rf"\date{{Generated {fetched_at} from W\&B project \texttt{{{project}}}}}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Experimental Setup}",
        "",
        r"Offline \texttt{ensemble-eval} on Stable Diffusion 3.5 Medium: multiple",
        r"FlowGRPO LoRA checkpoints are fused at each denoising step, then evaluated",
        r"on three held-out test sets (\texttt{geneval}, \texttt{pickscore}, \texttt{ocr}).",
        r"Checkpoints: \texttt{jieliu/SD3.5M-FlowGRPO-Text} (OCR),",
        r"\texttt{jieliu/SD3.5M-FlowGRPO-PickScore}, \texttt{jieliu/SD3.5M-FlowGRPO-GenEval}.",
        r"Rows with $n\geq 2$ teachers use uniform checkpoint averaging (\textsc{Avg})",
        r"unless marked \textsc{PCGrad}. Table~1: \textbf{bold} = single-teacher",
        r"in-domain metric; gray = in-domain for multi-teacher rows; red/blue",
        r"footnotes = change vs.\ the best single-teacher baseline among teachers",
        r"in that row (max over \texttt{1\_geneval}, \texttt{1\_ocr}, \texttt{1\_pickscore}",
        r"present in the ensemble).",
        r"Table~2 gray cells mark in-domain metrics (appendix).",
        "",
        r"\section{Main Results}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cross-benchmark eval metrics (mean over test set). "
        r"\textbf{Bold}: single-teacher in-domain. "
        r"Gray + red/blue footnotes: multi-teacher vs.\ max single-teacher in row.}",
        r"\label{tab:ensemble-eval-main}",
        r"\small",
        r"\begin{tabular}{ll|cc|c|cc}",
        r"\toprule",
        r" &  & \multicolumn{2}{c}{GenEval benchmark} "
        r"& \multicolumn{1}{c}{PickScore benchmark} "
        r"& \multicolumn{2}{c}{OCR benchmark} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7}",
        r"Teachers & Agg. & GenEval & PickScore & PickScore & OCR & PickScore \\",
        r"\midrule",
        *main_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\section{Full Eval Metrics}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{All scalar \texttt{eval/} metrics (std and GenEval per-tag breakdowns).}",
        r"\label{tab:ensemble-eval-appendix}",
        r"\footnotesize",
        rf"\resizebox{{\textwidth}}{{!}}{{%",
        rf"\begin{{tabular}}{{{appendix_col_spec}}}",
        r"\toprule",
        appendix_row1,
        appendix_bench_cmid,
        appendix_row2,
        appendix_reward_cmid,
        appendix_row3,
        r"\midrule",
        *appendix_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
        r"\end{document}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="W&B run names in table order",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".scratch/ensemble_eval_metrics.json"),
    )
    parser.add_argument(
        "--write-tex",
        type=Path,
        default=Path("docs/experiments.tex"),
    )
    args = parser.parse_args()

    api = wandb.Api()
    metrics_by_run: Dict[str, Dict[str, float]] = {}
    for run_name in args.runs:
        metrics_by_run[run_name] = fetch_run_metrics(api, args.project, run_name)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload: Dict[str, Any] = {
        "project": args.project,
        "fetched_at": fetched_at,
        "runs": args.runs,
        "metrics_by_run": metrics_by_run,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tex = generate_experiments_tex(
        project=args.project,
        runs=args.runs,
        metrics_by_run=metrics_by_run,
        fetched_at=fetched_at,
    )
    args.write_tex.parent.mkdir(parents=True, exist_ok=True)
    args.write_tex.write_text(tex, encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.write_tex}")


if __name__ == "__main__":
    main()
