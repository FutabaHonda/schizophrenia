#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create the nine NEURO2026 Results figures.

Outputs
-------
Three baseline bar charts (HC, Spine, DA, SZ):

1. Aberrant salience score
2. Initial discriminations achieved
3. Total reversals achieved

Six one-parameter-at-a-time sensitivity line charts (four group lines):

4. Aberrant salience score vs spine pathology strength
5. Aberrant salience score vs negative learning scale
6. Initial discriminations vs spine pathology strength
7. Initial discriminations vs negative learning scale
8. Total reversals vs spine pathology strength
9. Total reversals vs negative learning scale

The script imports the current ISP and Waltz model files and does not modify
them. The same agent seeds are reused at every value of a swept parameter
(common random numbers), so changes along a line primarily reflect the
parameter change rather than a different Monte Carlo sample.

Definition of ``spine pathology strength``
------------------------------------------
This is a dimensionless composite sensitivity parameter. A value of 1.0
reproduces the current pathological settings. The following model components
are scaled together:

* small-spine loss rate: 0.25 * strength
* XL-spine cluster SD: interpolated from HC=18 to pathological=8
* pathological XL/small mapping bias: interpolated from 0.50 to 0.80

The default sweep is centered around the fitted/current value (1.0). It is a
robustness analysis, not a newly fitted biological parameter.

Example
-------
python run_neuro2026_results_and_sensitivity.py \
  --isp_model EMBC_isp.py \
  --waltz_model EMBC_waltz.py \
  --out_dir neuro2026_results_sensitivity \
  --n_agents 50

Quick smoke test
----------------
python run_neuro2026_results_and_sensitivity.py \
  --isp_model EMBC_isp.py \
  --waltz_model EMBC_waltz.py \
  --out_dir smoke_test --n_agents 2 --n_spines 400 \
  --spine_strength_grid 0.5 1.0 \
  --negative_learning_grid 0.35 1.0 --no_svg
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-neuro2026")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUP_ORDER = ["HC", "Spine", "DA", "SZ"]
GROUP_COLORS = {
    "HC": "#1F77B4",
    "Spine": "#56B4E9",
    "DA": "#F2B01E",
    "SZ": "#E66B0A",
}
GROUP_HATCHES = {"HC": "", "Spine": "..", "DA": "//", "SZ": "xx"}

METRIC_INFO: Mapping[str, Dict[str, Any]] = {
    "aberrant_salience": {
        "model": "ISP",
        "column": "kat_aberrant_salience_ms",
        "title": "Aberrant salience score",
        "ylabel": "Aberrant salience score (ms)",
        "ylim": None,
        "number_format": ".1f",
    },
    "initial_discriminations": {
        "model": "Waltz",
        "column": "initial_discriminations",
        "title": "Initial discriminations achieved",
        "ylabel": "Discriminations achieved (0-3)",
        "ylim": (0.0, 3.25),
        "number_format": ".2f",
    },
    "total_reversals": {
        "model": "Waltz",
        "column": "total_reversals",
        "title": "Total reversals achieved",
        "ylabel": "Reversals achieved (0-6)",
        "ylim": (0.0, 6.35),
        "number_format": ".2f",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model: {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses in dynamically imported modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_model_path(requested: str, kind: str) -> Path:
    """Resolve explicit paths first, then common local/uploaded filenames."""
    p = Path(requested).expanduser()
    if p.exists():
        return p.resolve()

    cwd = Path.cwd()
    direct = cwd / p
    if direct.exists():
        return direct.resolve()

    if kind == "isp":
        patterns = [
            "EMBC_isp.py",
            "EMBC_isp*.py",
        ]
    else:
        patterns = [
            "EMBC_waltz.py",
            "EMBC_waltz*.py",
        ]

    candidates: List[Path] = []
    for parent in [cwd, cwd / "upload"]:
        for pattern in patterns:
            candidates.extend(sorted(parent.glob(pattern)))
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if len(unique) == 1:
        return unique[0]
    if unique:
        names = "\n  ".join(str(x) for x in unique)
        raise FileNotFoundError(
            f"Multiple {kind} model candidates were found. Specify --{kind}_model:\n  {names}"
        )
    raise FileNotFoundError(
        f"Could not find {kind} model '{requested}'. Specify --{kind}_model explicitly."
    )


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def spine_settings(strength: float) -> Dict[str, float]:
    """Return composite pathological settings; strength=1 matches baseline."""
    s = float(strength)
    return {
        "small_loss_rate": max(0.0, 0.25 * s),
        "xl_cluster_std": max(1.0, 18.0 + s * (8.0 - 18.0)),
        "mapping_bias": clip(0.50 + s * (0.80 - 0.50), 0.01, 0.99),
    }


def build_isp_components(model, args, strength: float, negative_scale: float):
    settings = spine_settings(strength)
    dend = model.DendriteParams(n_segments=args.n_segments)
    task = model.ISPTaskParams()
    fmap = model.FeatureMappingParams(
        active_frac_each_feature=args.active_frac_each_feature,
        xl_bias_pathological=settings["mapping_bias"],
        small_bias_pathological=settings["mapping_bias"],
    )
    nmda = model.NMDAGateParams(
        theta_quantile=args.theta_quantile,
        nmda_gain=args.nmda_gain,
        use_nmda=not args.no_nmda,
    )
    rtp = model.RTParams(
        beta0=args.beta0,
        beta_abs_pe=args.beta_abs_pe,
        beta_circle=args.beta_circle,
        beta_irrel_salience=args.beta_irrel_salience,
        beta_irrel_input=args.beta_irrel_input,
        rt_noise=args.rt_noise,
    )
    sp_hc = model.SpineParams(
        n_spines=args.n_spines,
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        small_loss_rate=0.0,
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
        xl_cluster_std=args.xl_cluster_std_hc,
    )
    sp_path = model.SpineParams(
        n_spines=args.n_spines,
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        small_loss_rate=settings["small_loss_rate"],
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
        xl_cluster_std=settings["xl_cluster_std"],
    )
    lp_hc = model.LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=1.0,
        init_weight=args.init_weight,
        bias=args.prediction_bias,
        weight_clip=args.weight_clip,
    )
    lp_da = model.LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=float(negative_scale),
        init_weight=args.init_weight,
        bias=args.prediction_bias,
        weight_clip=args.weight_clip,
    )

    # At exactly zero, make the spine factor an ablation identical in structure
    # to the healthy condition. The default sweep begins above zero.
    path_group = "healthy" if strength <= 0.0 else "sz"
    path_mapping = "healthy_like" if strength <= 0.0 else "pathological_like"
    path_spines = sp_hc if strength <= 0.0 else sp_path
    groups = [
        model.GroupConfig("HC", "healthy", 0, 0, sp_hc, lp_hc, "healthy_like"),
        model.GroupConfig("Spine", path_group, 1, 0, path_spines, lp_hc, path_mapping),
        model.GroupConfig("DA", "healthy", 0, 1, sp_hc, lp_da, "healthy_like"),
        model.GroupConfig("SZ", path_group, 1, 1, path_spines, lp_da, path_mapping),
    ]
    return dend, task, fmap, nmda, rtp, groups


def build_waltz_components(model, args, strength: float, negative_scale: float):
    settings = spine_settings(strength)
    dend = model.DendriteParams(n_segments=args.n_segments)
    mapping = model.MappingParams(
        active_frac_each_option=args.active_frac_each_option,
        healthy_xl_bias_A=0.50,
        healthy_xl_bias_B=0.50,
        pathological_xl_bias_A=settings["mapping_bias"],
        pathological_small_bias_B=settings["mapping_bias"],
        randomize_pathological_xl_option=args.randomize_pathological_xl_option,
    )
    task = model.WaltzTaskParams(
        n_pairs=3,
        n_reversals_max_per_pair=2,
        max_trials_per_stage=50,
        criterion_window=10,
        criterion_correct=9,
        p_correct_reward=0.80,
        p_incorrect_reward=0.20,
    )
    nmda = model.NMDAGateParams(
        use_nmda=not args.no_nmda,
        input_mode=args.waltz_input_mode,
        theta_quantile=args.theta_quantile,
        nmda_gain=args.nmda_gain,
    )
    sp_hc = model.SpineParams(
        n_spines=args.n_spines,
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        small_loss_rate=0.0,
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
        xl_cluster_std=args.xl_cluster_std_hc,
    )
    sp_path = model.SpineParams(
        n_spines=args.n_spines,
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        small_loss_rate=settings["small_loss_rate"],
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
        xl_cluster_std=settings["xl_cluster_std"],
    )
    lp_hc = model.LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=1.0,
        prediction_bias=args.prediction_bias,
        init_weight=args.init_weight,
        weight_clip=args.weight_clip,
        beta_choice=args.beta_choice,
        choice_value_noise=args.choice_value_noise,
    )
    lp_da = model.LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=float(negative_scale),
        prediction_bias=args.prediction_bias,
        init_weight=args.init_weight,
        weight_clip=args.weight_clip,
        beta_choice=args.beta_choice,
        choice_value_noise=args.choice_value_noise,
    )
    path_group = "healthy" if strength <= 0.0 else "sz"
    path_mapping = "healthy_like" if strength <= 0.0 else "pathological_like"
    path_spines = sp_hc if strength <= 0.0 else sp_path
    groups = [
        model.GroupConfig("HC", "healthy", "healthy_like", 0, 0, sp_hc, lp_hc),
        model.GroupConfig("Spine", path_group, path_mapping, 1, 0, path_spines, lp_hc),
        model.GroupConfig("DA", "healthy", "healthy_like", 0, 1, sp_hc, lp_da),
        model.GroupConfig("SZ", path_group, path_mapping, 1, 1, path_spines, lp_da),
    ]
    return dend, mapping, task, nmda, groups


def run_isp_condition(model, args, strength: float, negative_scale: float):
    dend, task, fmap, nmda, rtp, groups = build_isp_components(
        model, args, strength, negative_scale
    )
    rows: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups):
        group_rows, _ = model.simulate_one_group(
            group=group,
            n_agents=args.n_agents,
            base_seed=args.seed + 10_000_000 * (gi + 1),
            dend=dend,
            task=task,
            fmap_params=fmap,
            nmda=nmda,
            rtp=rtp,
            input_mode=args.isp_input_mode,
            save_histories=False,
            out_dir=str(args.out_dir),
        )
        rows.extend(group_rows)
    return rows


def run_waltz_condition(model, args, strength: float, negative_scale: float):
    dend, mapping, task, nmda, groups = build_waltz_components(
        model, args, strength, negative_scale
    )
    rows: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups):
        group_rows, _ = model.simulate_one_group(
            group,
            args.n_agents,
            args.seed + 10_000_000 * (gi + 1),
            dend,
            mapping,
            task,
            nmda,
            False,
            str(args.out_dir),
        )
        rows.extend(group_rows)
    return rows


def finite_values(rows: Sequence[Mapping[str, Any]], group: str, column: str):
    values = np.asarray(
        [float(r[column]) for r in rows if r["group_name"] == group], dtype=float
    )
    return values[np.isfinite(values)]


def normal_ci(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Return mean, SD, SEM, and an approximate 95% CI half-width."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(np.std(values, ddof=1))
    sem = sd / math.sqrt(values.size)
    return mean, sd, sem, 1.96 * sem


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, png_path: Path, save_svg: bool) -> None:
    """Save with a transparent outer canvas and white plotting areas.

    The title, axis labels, and tick labels are placed on the transparent
    Figure canvas, while the interior of each Axes remains white.
    """
    ensure_dir(png_path.parent)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",
        edgecolor="none",
    )
    if save_svg:
        fig.savefig(
            png_path.with_suffix(".svg"),
            bbox_inches="tight",
            facecolor="none",
            edgecolor="none",
        )
    plt.close(fig)


def style_axis(ax) -> None:
    # Keep only the graph interior opaque; the surrounding Figure is transparent.
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=14)


def plot_bar(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    out_path: Path,
    save_svg: bool,
    seed: int,
) -> None:
    info = METRIC_INFO[metric_key]
    column = info["column"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(GROUP_ORDER), dtype=float)
    means, ci95 = [], []
    for group in GROUP_ORDER:
        vals = finite_values(rows, group, column)
        mean, _, _, ci = normal_ci(vals)
        means.append(mean)
        ci95.append(ci)
    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=[GROUP_COLORS[g] for g in GROUP_ORDER],
        edgecolor="black",
        linewidth=1.4,
        yerr=ci95,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    for bar, group in zip(bars, GROUP_ORDER):
        bar.set_hatch(GROUP_HATCHES[group])

    rng = np.random.default_rng(seed)
    for gi, group in enumerate(GROUP_ORDER):
        vals = finite_values(rows, group, column)
        jitter = rng.uniform(-0.16, 0.16, size=vals.size)
        ax.scatter(
            np.full(vals.size, gi) + jitter,
            vals,
            s=18,
            color="black",
            alpha=0.24,
            linewidths=0,
            zorder=3,
        )
        mean = means[gi]
        ci = 0.0 if not np.isfinite(ci95[gi]) else ci95[gi]
        ax.text(
            gi,
            mean + ci + (0.03 * max(1.0, max(means))),
            format(mean, info["number_format"]),
            ha="center",
            va="bottom",
            fontsize=21,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=23, fontweight="bold")
    ax.set_ylabel(info["ylabel"], fontsize=24, fontweight="bold")
    ax.set_title(info["title"], fontsize=28, fontweight="bold", pad=14)
    if info["ylim"] is not None:
        ax.set_ylim(*info["ylim"])
    else:
        ymax = max(m + (c if np.isfinite(c) else 0.0) for m, c in zip(means, ci95))
        ax.set_ylim(0.0, max(1.0, ymax * 1.18))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_path, save_svg)


def plot_sensitivity(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    parameter: str,
    baseline_value: float,
    out_path: Path,
    save_svg: bool,
) -> None:
    info = METRIC_INFO[metric_key]
    parameter_label = (
        "Spine pathology strength"
        if parameter == "spine_pathology_strength"
        else "Negative learning scale (lower = more impaired)"
    )
    title_parameter = (
        "spine pathology strength"
        if parameter == "spine_pathology_strength"
        else "negative learning scale"
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    all_upper = []
    for group in GROUP_ORDER:
        sub = [
            r
            for r in summary_rows
            if r["metric_key"] == metric_key
            and r["parameter"] == parameter
            and r["group_name"] == group
        ]
        sub = sorted(sub, key=lambda r: float(r["parameter_value"]))
        x = np.asarray([float(r["parameter_value"]) for r in sub], dtype=float)
        y = np.asarray([float(r["mean"]) for r in sub], dtype=float)
        ci = np.asarray([float(r["ci95_halfwidth"]) for r in sub], dtype=float)
        ci = np.where(np.isfinite(ci), ci, 0.0)
        all_upper.extend((y + ci).tolist())
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=GROUP_COLORS[group],
            label=group,
        )
        ax.fill_between(x, y - ci, y + ci, color=GROUP_COLORS[group], alpha=0.11)

    ax.axvline(
        baseline_value,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label="Baseline",
        zorder=1,
    )
    ax.set_xlabel(parameter_label, fontsize=23, fontweight="bold")
    sensitivity_ylabel = {
        "aberrant_salience": info["ylabel"],
        "initial_discriminations": "Discriminations achieved",
        "total_reversals": "Reversals achieved",
    }[metric_key]
    ax.set_ylabel(sensitivity_ylabel, fontsize=24, fontweight="bold")
    ax.set_title(
        f"Vs {title_parameter}",
        fontsize=27,
        fontweight="bold",
        pad=12,
    )
    if info["ylim"] is not None:
        ax.set_ylim(*info["ylim"])
    else:
        ymax = max(all_upper) if all_upper else 1.0
        ax.set_ylim(0.0, max(1.0, ymax * 1.12))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_path, save_svg)


def tag_agent_rows(
    rows: Sequence[Mapping[str, Any]],
    model_name: str,
    parameter: str | None = None,
    parameter_value: float | None = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        tagged = dict(row)
        tagged["source_model"] = model_name
        if parameter is not None:
            tagged["parameter"] = parameter
            tagged["parameter_value"] = parameter_value
        out.append(tagged)
    return out


def summarize_condition(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    parameter: str,
    parameter_value: float,
) -> List[Dict[str, Any]]:
    info = METRIC_INFO[metric_key]
    column = info["column"]
    out = []
    for group in GROUP_ORDER:
        vals = finite_values(rows, group, column)
        mean, sd, sem, ci = normal_ci(vals)
        out.append(
            {
                "parameter": parameter,
                "parameter_value": float(parameter_value),
                "metric_key": metric_key,
                "metric_column": column,
                "source_model": info["model"],
                "group_name": group,
                "n": int(vals.size),
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "ci95_halfwidth": ci,
                "ci95_low": mean - ci if np.isfinite(ci) else float("nan"),
                "ci95_high": mean + ci if np.isfinite(ci) else float("nan"),
            }
        )
    return out


def unique_sorted(values: Iterable[float]) -> List[float]:
    return sorted({float(v) for v in values})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate three baseline Results bars and six four-group sensitivity curves."
    )
    ap.add_argument(
        "--isp_model",
        default="EMBC_isp.py",
        help="Path to the current ISP model script.",
    )
    ap.add_argument(
        "--waltz_model",
        default="EMBC_waltz.py",
        help="Path to the current reversal-learning model script.",
    )
    ap.add_argument(
        "--out_dir", type=Path, default=Path("neuro2026_results_sensitivity")
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_agents", type=int, default=50)
    ap.add_argument("--n_spines", type=int, default=4000)
    ap.add_argument("--n_segments", type=int, default=80)
    ap.add_argument(
        "--spine_strength_grid",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.00, 1.25],
    )
    ap.add_argument(
        "--negative_learning_grid",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.35, 0.50, 0.70, 1.00],
    )
    ap.add_argument("--baseline_spine_strength", type=float, default=1.0)
    ap.add_argument("--baseline_negative_learning_scale", type=float, default=0.35)
    ap.add_argument("--no_svg", action="store_true")

    # Shared baseline parameters from the current model files.
    ap.add_argument("--lognorm_mu", type=float, default=-0.5)
    ap.add_argument("--lognorm_sigma", type=float, default=0.45)
    ap.add_argument("--xl_size_threshold", type=float, default=0.8)
    ap.add_argument("--small_loss_threshold", type=float, default=0.25)
    ap.add_argument("--xl_extra_mu", type=float, default=0.0)
    ap.add_argument("--xl_extra_sigma", type=float, default=0.22)
    ap.add_argument("--xl_cluster_std_hc", type=float, default=18.0)
    ap.add_argument("--nmda_gain", type=float, default=3.5)
    ap.add_argument("--theta_quantile", type=float, default=0.92)
    ap.add_argument("--no_nmda", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--init_weight", type=float, default=0.0)
    ap.add_argument("--prediction_bias", type=float, default=0.0)
    ap.add_argument("--weight_clip", type=float, default=5.0)

    # ISP-specific settings.
    ap.add_argument(
        "--isp_input_mode", choices=["linear", "nmda_excess", "total"], default="total"
    )
    ap.add_argument("--active_frac_each_feature", type=float, default=0.015)
    ap.add_argument("--beta0", type=float, default=6.18)
    ap.add_argument("--beta_abs_pe", type=float, default=0.35)
    ap.add_argument("--beta_circle", type=float, default=0.05)
    ap.add_argument("--beta_irrel_salience", type=float, default=0.20)
    ap.add_argument("--beta_irrel_input", type=float, default=0.03)
    ap.add_argument("--rt_noise", type=float, default=0.02)

    # Waltz-specific settings.
    ap.add_argument(
        "--waltz_input_mode", choices=["linear", "nmda", "total"], default="total"
    )
    ap.add_argument("--active_frac_each_option", type=float, default=0.05)
    ap.add_argument("--beta_choice", type=float, default=5.0)
    ap.add_argument("--choice_value_noise", type=float, default=0.0)
    ap.add_argument("--randomize_pathological_xl_option", action="store_true")
    args = ap.parse_args()

    if args.n_agents < 1:
        ap.error("--n_agents must be >= 1")
    if args.n_spines < 20:
        ap.error("--n_spines must be >= 20")
    if args.n_segments < 2:
        ap.error("--n_segments must be >= 2")
    if not 0.0 < args.baseline_negative_learning_scale <= 1.0:
        ap.error("--baseline_negative_learning_scale must be in (0, 1]")
    if any(v <= 0.0 for v in args.negative_learning_grid):
        ap.error("All --negative_learning_grid values must be > 0")

    args.out_dir = args.out_dir.resolve()
    ensure_dir(args.out_dir)
    bar_dir = args.out_dir / "bars"
    sensitivity_dir = args.out_dir / "sensitivity"
    data_dir = args.out_dir / "data"
    ensure_dir(bar_dir)
    ensure_dir(sensitivity_dir)
    ensure_dir(data_dir)

    isp_path = resolve_model_path(args.isp_model, "isp")
    waltz_path = resolve_model_path(args.waltz_model, "waltz")
    isp_model = load_module(isp_path, "neuro2026_isp_model")
    waltz_model = load_module(waltz_path, "neuro2026_waltz_model")

    spine_grid = unique_sorted(
        list(args.spine_strength_grid) + [args.baseline_spine_strength]
    )
    neg_grid = unique_sorted(
        list(args.negative_learning_grid) + [args.baseline_negative_learning_scale]
    )

    print("=== NEURO2026 Results + sensitivity ===")
    print(f"ISP model: {isp_path}")
    print(f"Waltz model: {waltz_path}")
    print(f"Agents per group: {args.n_agents}")
    print(f"Spine strength grid: {spine_grid}")
    print(f"Negative learning grid: {neg_grid}")

    isp_cache: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    waltz_cache: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}

    def cache_key(s: float, n: float) -> Tuple[float, float]:
        return round(float(s), 10), round(float(n), 10)

    def get_isp(s: float, n: float):
        key = cache_key(s, n)
        if key not in isp_cache:
            print(f"[ISP] spine={s:g}, negative_scale={n:g}")
            isp_cache[key] = run_isp_condition(isp_model, args, s, n)
        return isp_cache[key]

    def get_waltz(s: float, n: float):
        key = cache_key(s, n)
        if key not in waltz_cache:
            print(f"[Waltz] spine={s:g}, negative_scale={n:g}")
            waltz_cache[key] = run_waltz_condition(waltz_model, args, s, n)
        return waltz_cache[key]

    # Run each model once per unique condition. Baseline conditions are cached
    # and reused for the bar charts and both sensitivity analyses.
    for value in spine_grid:
        get_isp(value, args.baseline_negative_learning_scale)
        get_waltz(value, args.baseline_negative_learning_scale)
    for value in neg_grid:
        get_isp(args.baseline_spine_strength, value)
        get_waltz(args.baseline_spine_strength, value)

    baseline_isp = get_isp(
        args.baseline_spine_strength, args.baseline_negative_learning_scale
    )
    baseline_waltz = get_waltz(
        args.baseline_spine_strength, args.baseline_negative_learning_scale
    )

    baseline_agents = tag_agent_rows(baseline_isp, "ISP") + tag_agent_rows(
        baseline_waltz, "Waltz"
    )
    write_csv(data_dir / "baseline_agent_metrics.csv", baseline_agents)

    sensitivity_agents: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for value in spine_grid:
        isp_rows = get_isp(value, args.baseline_negative_learning_scale)
        waltz_rows = get_waltz(value, args.baseline_negative_learning_scale)
        sensitivity_agents.extend(
            tag_agent_rows(isp_rows, "ISP", "spine_pathology_strength", float(value))
        )
        sensitivity_agents.extend(
            tag_agent_rows(
                waltz_rows, "Waltz", "spine_pathology_strength", float(value)
            )
        )
        summary_rows.extend(
            summarize_condition(
                isp_rows,
                "aberrant_salience",
                "spine_pathology_strength",
                value,
            )
        )
        for metric_key in ["initial_discriminations", "total_reversals"]:
            summary_rows.extend(
                summarize_condition(
                    waltz_rows,
                    metric_key,
                    "spine_pathology_strength",
                    value,
                )
            )

    for value in neg_grid:
        isp_rows = get_isp(args.baseline_spine_strength, value)
        waltz_rows = get_waltz(args.baseline_spine_strength, value)
        sensitivity_agents.extend(
            tag_agent_rows(isp_rows, "ISP", "negative_learning_scale", float(value))
        )
        sensitivity_agents.extend(
            tag_agent_rows(waltz_rows, "Waltz", "negative_learning_scale", float(value))
        )
        summary_rows.extend(
            summarize_condition(
                isp_rows,
                "aberrant_salience",
                "negative_learning_scale",
                value,
            )
        )
        for metric_key in ["initial_discriminations", "total_reversals"]:
            summary_rows.extend(
                summarize_condition(
                    waltz_rows,
                    metric_key,
                    "negative_learning_scale",
                    value,
                )
            )

    write_csv(data_dir / "sensitivity_agent_metrics.csv", sensitivity_agents)
    write_csv(data_dir / "sensitivity_group_summary.csv", summary_rows)

    save_svg = not args.no_svg
    plot_bar(
        baseline_isp,
        "aberrant_salience",
        bar_dir / "01_aberrant_salience_score.png",
        save_svg,
        args.seed + 101,
    )
    plot_bar(
        baseline_waltz,
        "initial_discriminations",
        bar_dir / "02_initial_discriminations_achieved.png",
        save_svg,
        args.seed + 102,
    )
    plot_bar(
        baseline_waltz,
        "total_reversals",
        bar_dir / "03_total_reversals_achieved.png",
        save_svg,
        args.seed + 103,
    )

    line_specs = [
        (
            "aberrant_salience",
            "spine_pathology_strength",
            args.baseline_spine_strength,
            "04_aberrant_salience_by_spine_strength.png",
        ),
        (
            "aberrant_salience",
            "negative_learning_scale",
            args.baseline_negative_learning_scale,
            "05_aberrant_salience_by_negative_learning.png",
        ),
        (
            "initial_discriminations",
            "spine_pathology_strength",
            args.baseline_spine_strength,
            "06_initial_discriminations_by_spine_strength.png",
        ),
        (
            "initial_discriminations",
            "negative_learning_scale",
            args.baseline_negative_learning_scale,
            "07_initial_discriminations_by_negative_learning.png",
        ),
        (
            "total_reversals",
            "spine_pathology_strength",
            args.baseline_spine_strength,
            "08_total_reversals_by_spine_strength.png",
        ),
        (
            "total_reversals",
            "negative_learning_scale",
            args.baseline_negative_learning_scale,
            "09_total_reversals_by_negative_learning.png",
        ),
    ]
    for metric_key, parameter, baseline, filename in line_specs:
        plot_sensitivity(
            summary_rows,
            metric_key,
            parameter,
            baseline,
            sensitivity_dir / filename,
            save_svg,
        )

    settings = {
        "isp_model": str(isp_path),
        "waltz_model": str(waltz_path),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "n_agents": args.n_agents,
        "n_spines": args.n_spines,
        "n_segments": args.n_segments,
        "baseline_spine_strength": args.baseline_spine_strength,
        "baseline_negative_learning_scale": args.baseline_negative_learning_scale,
        "spine_strength_grid": spine_grid,
        "negative_learning_grid": neg_grid,
        "spine_strength_definition": {
            "small_loss_rate": "0.25 * strength",
            "xl_cluster_std": "18 + strength * (8 - 18), minimum 1",
            "mapping_bias": "0.50 + strength * (0.80 - 0.50), clipped to [0.01, 0.99]",
        },
        "uncertainty": "mean +/- 1.96 SEM",
        "common_random_numbers": True,
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    with (data_dir / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print("\nSaved nine figures:")
    for path in sorted(bar_dir.glob("*.png")) + sorted(sensitivity_dir.glob("*.png")):
        print(path)
    print("\nSaved data:")
    for path in sorted(data_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
