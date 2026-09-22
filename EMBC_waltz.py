#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spine-weighted Waltz-like probabilistic reversal learning simulation.

Core mechanism is matched to the spine-weighted ISP model:
spine size distribution -> option-specific input x -> prediction Q -> prediction
error delta -> input-weighted learning -> reversal-learning behavior.

Groups:
  HC    healthy spine + normal negative learning
  Spine pathological spine + normal negative learning
  DA    healthy spine + impaired negative learning
  SZ    pathological spine + impaired negative learning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sigmoid(x: float) -> float:
    x = float(max(-60.0, min(60.0, x)))
    return 1.0 / (1.0 + math.exp(-x))


def clip_value(x: float, wmax: float) -> float:
    return float(max(-wmax, min(wmax, x)))


def safe_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def safe_sem(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, ci: float = 95.0
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        m = float(np.mean(values))
        return m, m
    n = values.size
    means = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=n, replace=True))
    a = (100.0 - ci) / 2.0
    return float(np.percentile(means, a)), float(np.percentile(means, 100.0 - a))


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class SpineParams:
    n_spines: int = 4000
    feat_dim: int = 2
    feat_range: float = 100.0
    lognorm_mu: float = -0.5
    lognorm_sigma: float = 0.45
    xl_size_threshold: float = 0.8
    small_loss_threshold: float = 0.25
    small_loss_rate: float = 0.25
    xl_extra_mu: float = 0.0
    xl_extra_sigma: float = 0.22
    xl_n_clusters: int = 12
    xl_cluster_std: float = 12.0


@dataclass
class DendriteParams:
    n_segments: int = 80
    kmeans_iters: int = 20


@dataclass
class MappingParams:
    active_frac_each_option: float = 0.05
    healthy_xl_bias_A: float = 0.50
    healthy_xl_bias_B: float = 0.50
    pathological_xl_bias_A: float = 0.80
    pathological_small_bias_B: float = 0.80
    randomize_pathological_xl_option: bool = False


@dataclass
class NMDAGateParams:
    use_nmda: bool = True
    input_mode: str = "total"  # linear, nmda, total
    theta_quantile: float = 0.92
    nmda_gain: float = 3.5


@dataclass
class LearningParams:
    alpha: float = 0.04
    alpha_neg_scale: float = 1.0
    prediction_bias: float = 0.0
    init_weight: float = 0.0
    weight_clip: float = 5.0
    beta_choice: float = 5.0
    choice_value_noise: float = 0.0


@dataclass
class WaltzTaskParams:
    n_pairs: int = 3
    n_reversals_max_per_pair: int = 2
    max_trials_per_stage: int = 50
    criterion_window: int = 10
    criterion_correct: int = 9
    p_correct_reward: float = 0.80
    p_incorrect_reward: float = 0.20


@dataclass
class GroupConfig:
    group_name: str
    spine_group: str
    mapping_mode: str
    spine_factor: int
    da_factor: int
    spine_params: SpineParams
    learning_params: LearningParams


def generate_spines(
    rng: np.random.Generator, sp: SpineParams, group: str
) -> Dict[str, np.ndarray]:
    n = sp.n_spines
    sizes = rng.lognormal(mean=sp.lognorm_mu, sigma=sp.lognorm_sigma, size=n)
    if group == "sz":
        small_idx = np.where(sizes < sp.small_loss_threshold)[0]
        keep = np.ones(n, dtype=bool)
        n_drop = int(round(sp.small_loss_rate * small_idx.size))
        if n_drop > 0 and small_idx.size > 0:
            keep[rng.choice(small_idx, size=n_drop, replace=False)] = False
        surviving = sizes[keep]
        n_missing = n - surviving.size
        if n_missing > 0:
            extra = rng.lognormal(
                mean=sp.xl_extra_mu, sigma=sp.xl_extra_sigma, size=n_missing
            )
            sizes = np.concatenate([surviving, extra])
        else:
            sizes = surviving[:n].copy()
        rng.shuffle(sizes)
    elif group != "healthy":
        raise ValueError("group must be healthy or sz")

    xl_mask = sizes >= sp.xl_size_threshold
    small_mask = sizes <= sp.small_loss_threshold
    pos = np.empty((n, sp.feat_dim), dtype=float)
    non_idx = np.where(~xl_mask)[0]
    pos[non_idx] = rng.uniform(
        -sp.feat_range, sp.feat_range, size=(non_idx.size, sp.feat_dim)
    )
    xl_idx = np.where(xl_mask)[0]
    if xl_idx.size > 0:
        centers = rng.uniform(
            -sp.feat_range, sp.feat_range, size=(sp.xl_n_clusters, sp.feat_dim)
        )
        rng.shuffle(xl_idx)
        for c, ids in enumerate(np.array_split(xl_idx, sp.xl_n_clusters)):
            if ids.size:
                pos[ids] = centers[c] + rng.normal(
                    0.0, sp.xl_cluster_std, size=(ids.size, sp.feat_dim)
                )
    return {
        "sizes": sizes.astype(float),
        "weights": sizes.astype(float),
        "pos": pos,
        "xl_mask": xl_mask,
        "small_mask": small_mask,
    }


def kmeans_segments(
    pos: np.ndarray, n_segments: int, iters: int, rng: np.random.Generator
) -> np.ndarray:
    n = pos.shape[0]
    k = min(n_segments, n)
    centers = pos[rng.choice(np.arange(n), size=k, replace=False)].copy()
    seg = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((pos[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        seg = np.argmin(d2, axis=1)
        for j in range(k):
            idx = np.where(seg == j)[0]
            centers[j] = pos[idx].mean(axis=0) if idx.size else pos[rng.integers(0, n)]
    return seg


def sample_mix(rng, n, preferred, other, active_k, pref_frac):
    pref_frac = float(max(0.0, min(1.0, pref_frac)))
    k_pref = int(round(active_k * pref_frac))
    k_other = active_k - k_pref
    chosen = []
    if k_pref > 0 and preferred.size > 0:
        chosen.append(
            rng.choice(preferred, size=min(k_pref, preferred.size), replace=False)
        )
    if k_other > 0 and other.size > 0:
        chosen.append(rng.choice(other, size=min(k_other, other.size), replace=False))
    if chosen:
        idx = np.unique(np.concatenate(chosen))
        if idx.size < active_k:
            rest = np.setdiff1d(np.arange(n), idx)
            idx = np.concatenate(
                [idx, rng.choice(rest, size=active_k - idx.size, replace=False)]
            )
        if idx.size > active_k:
            idx = rng.choice(idx, size=active_k, replace=False)
    else:
        idx = rng.choice(np.arange(n), size=active_k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def make_pair_option_spine_sets(
    rng, mapping_mode, n_pairs, active_k, xl_mask, small_mask, mp
):
    n = xl_mask.size
    xl_ids = np.where(xl_mask)[0]
    nonxl_ids = np.where(~xl_mask)[0]
    small_ids = np.where(small_mask)[0]
    nonsmall_ids = np.where(~small_mask)[0]
    out = {}
    for p in range(n_pairs):
        if mapping_mode == "healthy_like":
            A = sample_mix(rng, n, xl_ids, nonxl_ids, active_k, mp.healthy_xl_bias_A)
            B = sample_mix(rng, n, xl_ids, nonxl_ids, active_k, mp.healthy_xl_bias_B)
        elif mapping_mode == "pathological_like":
            xl_rich = sample_mix(
                rng, n, xl_ids, nonxl_ids, active_k, mp.pathological_xl_bias_A
            )
            small_rich = sample_mix(
                rng, n, small_ids, nonsmall_ids, active_k, mp.pathological_small_bias_B
            )
            xl_on_A = True
            if mp.randomize_pathological_xl_option:
                xl_on_A = bool(rng.integers(0, 2))
            A, B = (xl_rich, small_rich) if xl_on_A else (small_rich, xl_rich)
        else:
            raise ValueError("unknown mapping_mode")
        out[p] = {"A": A, "B": B}
    return out


def compute_pair_inputs(pair_maps, spine, seg_id, nmda):
    weights = spine["weights"]
    xl_mask = spine["xl_mask"]
    A_mask, B_mask = pair_maps["A"], pair_maps["B"]
    active = A_mask | B_mask
    n_segments = int(seg_id.max()) + 1
    seg_sums = np.zeros(n_segments, dtype=float)
    idx = np.where(active)[0]
    if idx.size:
        np.add.at(seg_sums, seg_id[idx], weights[idx])
    if nmda.use_nmda:
        theta = float(np.quantile(seg_sums, nmda.theta_quantile))
        fired_seg = seg_sums > theta
    else:
        theta = float("inf")
        fired_seg = np.zeros(n_segments, dtype=bool)

    def one(mask):
        ids = np.where(mask)[0]
        lin = float(np.sum(weights[ids])) if ids.size else 0.0
        if nmda.use_nmda and ids.size:
            fired = fired_seg[seg_id[ids]].astype(float)
            extra = float(np.sum(weights[ids] * (nmda.nmda_gain - 1.0) * fired))
        else:
            extra = 0.0
        total = lin + extra
        if nmda.input_mode == "linear":
            used = lin
        elif nmda.input_mode == "nmda":
            used = extra
        else:
            used = total
        return lin, extra, total, used

    A_lin, A_nmda, A_total, A_used = one(A_mask)
    B_lin, B_nmda, B_total, B_used = one(B_mask)
    mean_w = float(np.mean(weights) + 1e-12)
    x_A = A_used / (float(np.sum(A_mask)) * mean_w + 1e-12)
    x_B = B_used / (float(np.sum(B_mask)) * mean_w + 1e-12)
    fired_mask = active & fired_seg[seg_id]
    fired_xl_fraction = (
        float(np.mean(xl_mask[fired_mask])) if np.any(fired_mask) else 0.0
    )
    return {
        "x_A": float(x_A),
        "x_B": float(x_B),
        "input_A_linear": A_lin,
        "input_A_nmda": A_nmda,
        "input_A_total": A_total,
        "input_B_linear": B_lin,
        "input_B_nmda": B_nmda,
        "input_B_total": B_total,
        "fired_segment_count": float(np.sum(fired_seg)),
        "fired_xl_fraction": fired_xl_fraction,
        "A_xl_fraction": float(np.mean(xl_mask[A_mask])),
        "B_xl_fraction": float(np.mean(xl_mask[B_mask])),
    }


def correct_option_for_stage(stage_idx: int) -> str:
    return "A" if stage_idx % 2 == 0 else "B"


def choose_option(rng, Q_A, Q_B, beta_choice):
    zA, zB = beta_choice * Q_A, beta_choice * Q_B
    m = max(zA, zB)
    pA = math.exp(zA - m) / (math.exp(zA - m) + math.exp(zB - m))
    return "A" if rng.random() < pA else "B"


def simulate_waltz_agent(rng, spine, seg_id, pair_maps, task, nmda, learn):
    W = {
        p: {"A": float(learn.init_weight), "B": float(learn.init_weight)}
        for p in range(task.n_pairs)
    }
    trial_rows, pair_input_rows = [], []
    initial_achieved = first_rev_achieved = total_rev_achieved = 0
    disc_errors = disc_trials = firstrev_errors = firstrev_trials = 0
    perseverative_errors = perseverative_trials = 0

    for p in range(task.n_pairs):
        inp = compute_pair_inputs(pair_maps[p], spine, seg_id, nmda)
        pair_input_rows.append({"pair": p, **inp})
        x = {"A": inp["x_A"], "B": inp["x_B"]}
        n_stages = 1 + task.n_reversals_max_per_pair
        for stage in range(n_stages):
            correct = correct_option_for_stage(stage)
            recent = []
            achieved = False
            for trial_in_stage in range(task.max_trials_per_stage):
                Q_A = sigmoid(learn.prediction_bias + W[p]["A"] * x["A"])
                Q_B = sigmoid(learn.prediction_bias + W[p]["B"] * x["B"])
                QAc = (
                    Q_A + rng.normal(0.0, learn.choice_value_noise)
                    if learn.choice_value_noise > 0
                    else Q_A
                )
                QBc = (
                    Q_B + rng.normal(0.0, learn.choice_value_noise)
                    if learn.choice_value_noise > 0
                    else Q_B
                )
                choice = choose_option(rng, QAc, QBc, learn.beta_choice)
                is_correct = int(choice == correct)
                p_reward = (
                    task.p_correct_reward if is_correct else task.p_incorrect_reward
                )
                R = 1 if rng.random() < p_reward else 0
                Q_choice = Q_A if choice == "A" else Q_B
                delta = float(R - Q_choice)
                alpha_eff = (
                    learn.alpha if delta >= 0 else learn.alpha * learn.alpha_neg_scale
                )
                W_old = W[p][choice]
                W[p][choice] = clip_value(
                    W[p][choice] + alpha_eff * delta * x[choice], learn.weight_clip
                )

                if stage == 0:
                    disc_trials += 1
                    disc_errors += int(not is_correct)
                elif stage == 1:
                    firstrev_trials += 1
                    firstrev_errors += int(not is_correct)
                    perseverative_trials += 1
                    if choice == "A":
                        perseverative_errors += 1

                recent.append(is_correct)
                if len(recent) > task.criterion_window:
                    recent.pop(0)
                trial_rows.append(
                    {
                        "pair": p,
                        "stage": stage,
                        "trial_in_stage": trial_in_stage,
                        "correct_option": correct,
                        "choice": choice,
                        "is_correct": is_correct,
                        "reward": R,
                        "p_reward": p_reward,
                        "Q_A": Q_A,
                        "Q_B": Q_B,
                        "W_A": W[p]["A"],
                        "W_B": W[p]["B"],
                        "W_choice_old": W_old,
                        "delta": delta,
                        "alpha_eff": alpha_eff,
                        "x_A": x["A"],
                        "x_B": x["B"],
                        "x_choice": x[choice],
                        "fired_xl_fraction": inp["fired_xl_fraction"],
                    }
                )
                if (
                    len(recent) == task.criterion_window
                    and sum(recent) >= task.criterion_correct
                ):
                    achieved = True
                    break
            if achieved:
                if stage == 0:
                    initial_achieved += 1
                elif stage == 1:
                    first_rev_achieved += 1
                    total_rev_achieved += 1
                else:
                    total_rev_achieved += 1
            else:
                break

    deltas = np.array([r["delta"] for r in trial_rows], dtype=float)
    x_A_vals = np.array([r["x_A"] for r in pair_input_rows], dtype=float)
    x_B_vals = np.array([r["x_B"] for r in pair_input_rows], dtype=float)
    input_imb = np.abs(x_A_vals - x_B_vals)
    input_ratio = np.maximum(x_A_vals, x_B_vals) / (
        np.minimum(x_A_vals, x_B_vals) + 1e-12
    )
    summary = {
        "initial_discriminations": float(initial_achieved),
        "first_reversals": float(first_rev_achieved),
        "total_reversals": float(total_rev_achieved),
        "discrimination_error_rate": (
            float(disc_errors / disc_trials) if disc_trials else float("nan")
        ),
        "first_reversal_error_rate": (
            float(firstrev_errors / firstrev_trials)
            if firstrev_trials
            else float("nan")
        ),
        "perseverative_error_rate_firstrev": (
            float(perseverative_errors / perseverative_trials)
            if perseverative_trials
            else float("nan")
        ),
        "n_trials_total": float(len(trial_rows)),
        "mean_abs_pe": safe_mean(np.abs(deltas)),
        "mean_delta": safe_mean(deltas),
        "mean_x_A": safe_mean(x_A_vals),
        "mean_x_B": safe_mean(x_B_vals),
        "mean_option_input_imbalance": safe_mean(input_imb),
        "mean_option_input_ratio": safe_mean(input_ratio),
        "mean_fired_xl_fraction": safe_mean(
            [r["fired_xl_fraction"] for r in pair_input_rows]
        ),
        "mean_A_xl_fraction": safe_mean([r["A_xl_fraction"] for r in pair_input_rows]),
        "mean_B_xl_fraction": safe_mean([r["B_xl_fraction"] for r in pair_input_rows]),
        "mean_input_A_total": safe_mean([r["input_A_total"] for r in pair_input_rows]),
        "mean_input_B_total": safe_mean([r["input_B_total"] for r in pair_input_rows]),
    }
    return {
        "summary": summary,
        "trial_rows": trial_rows,
        "pair_input_rows": pair_input_rows,
    }


def group_summary(rows, metrics, n_boot, rng):
    out = []
    for g in ["HC", "Spine", "DA", "SZ"]:
        group_rows = [r for r in rows if r["group_name"] == g]
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in group_rows], dtype=float)
            lo, hi = bootstrap_ci(vals, rng, n_boot=n_boot)
            out.append(
                {
                    "group_name": g,
                    "metric": metric,
                    "n": len(group_rows),
                    "mean": safe_mean(vals),
                    "sd": safe_std(vals),
                    "sem": safe_sem(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return out


def fit_ols_sse(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if y.size == 0:
        return float("nan"), 0
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid**2)), int(y.size - np.linalg.matrix_rank(X))


def factorial_anova_one_metric(rows, metric, n_perm, rng):
    y = np.array([float(r[metric]) for r in rows], dtype=float)
    spine = np.array([int(r["spine_factor"]) for r in rows], dtype=float)
    da = np.array([int(r["da_factor"]) for r in rows], dtype=float)
    inter = spine * da
    X_full = np.column_stack([np.ones_like(spine), spine, da, inter])
    full_sse, full_df = fit_ols_sse(y, X_full)
    effects = {"Spine": [0, 2, 3], "DA": [0, 1, 3], "Spine_x_DA": [0, 1, 2]}
    out = {}
    for name, cols in effects.items():
        X_red = X_full[:, cols]
        red_sse, red_df = fit_ols_sse(y, X_red)
        df_num, df_den = red_df - full_df, full_df
        if df_num <= 0 or df_den <= 0 or np.isnan(red_sse) or np.isnan(full_sse):
            f_obs = float("nan")
        else:
            ms_num = (red_sse - full_sse) / df_num
            ms_den = full_sse / df_den
            f_obs = float(ms_num / ms_den) if ms_den > 0 else float("nan")
        p_param = (
            float(stats.f.sf(f_obs, df_num, df_den))
            if HAS_SCIPY and np.isfinite(f_obs)
            else float("nan")
        )
        p_perm = float("nan")
        if n_perm > 0 and np.isfinite(f_obs):
            count = 0
            for _ in range(n_perm):
                yp = rng.permutation(y)
                fsse, fdf = fit_ols_sse(yp, X_full)
                rsse, rdf = fit_ols_sse(yp, X_red)
                dfn, dfd = rdf - fdf, fdf
                if dfn <= 0 or dfd <= 0:
                    continue
                fperm = ((rsse - fsse) / dfn) / (fsse / dfd) if fsse > 0 else 0.0
                if fperm >= f_obs:
                    count += 1
            p_perm = float((count + 1) / (n_perm + 1))
        out[name] = {
            "F": f_obs,
            "df_num": float(df_num),
            "df_den": float(df_den),
            "p_parametric": p_param,
            "p_permutation": p_perm,
        }
    return out


def flatten_anova_rows(anova):
    return [
        {"metric": m, "effect": e, **vals}
        for m, eff in anova.items()
        for e, vals in eff.items()
    ]


GROUP_ORDER = ["HC", "Spine", "DA", "SZ"]
GROUP_COLORS = {
    "HC": "#1f77b4",  # blue
    "Spine": "#f2c300",  # yellow
    "DA": "#ff7f0e",  # orange
    "SZ": "#d62728",  # red
}

# Large-font settings for A0 poster figures.
TITLE_FS = 26
PANEL_FS = 28
AXIS_LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 18


def get_group_values(rows, group, metric):
    vals = np.array(
        [float(r[metric]) for r in rows if r["group_name"] == group], dtype=float
    )
    return vals[np.isfinite(vals)]


def plot_summary_metrics(rows, metrics, out_path):
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib is not available. Skipping plot.")
        return
    group_order = GROUP_ORDER
    n_cols = 2
    n_rows = int(math.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5.0 * n_rows), dpi=300)
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(12345)
    for ax, metric in zip(axes, metrics):
        means, sems = [], []
        for g in group_order:
            vals = get_group_values(rows, g, metric)
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))
        x = np.arange(len(group_order))
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[GROUP_COLORS[g] for g in group_order],
            alpha=0.90,
            edgecolor="black",
            linewidth=0.8,
            capsize=4,
        )
        for i, g in enumerate(group_order):
            vals = get_group_values(rows, g, metric)
            ax.scatter(
                np.full(vals.size, i) + rng.normal(0.0, 0.055, size=vals.size),
                vals,
                color="black",
                s=12,
                alpha=0.45,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, fontsize=TICK_FS)
        ax.set_title(metric, fontweight="bold", fontsize=AXIS_LABEL_FS)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_FS)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(
        "Spine-weighted Waltz reversal-learning simulation",
        fontweight="bold",
        fontsize=TITLE_FS,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _percent_distribution(rows, metric, group, categories, bin_func=None):
    vals = get_group_values(rows, group, metric)
    n = vals.size
    if n == 0:
        return np.full(len(categories), np.nan, dtype=float)
    out = []
    for cat in categories:
        if bin_func is None:
            count = np.sum(vals == float(cat))
        else:
            count = np.sum([bin_func(v) == cat for v in vals])
        out.append(100.0 * count / n)
    return np.array(out, dtype=float)


def plot_waltz_reference_style(rows, out_path):
    """Create a Waltz & Gold Fig. 1-like 2x2 bar plot for 4 model groups.

    A: Distribution of initial discriminations achieved.
    B: Distribution of total reversals achieved, binned as in Waltz & Gold.
    C: Distribution of first reversals achieved.
    D: Mean error rates for initial discrimination and first reversal.
    """
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib is not available. Skipping Waltz-style plot.")
        return

    group_order = GROUP_ORDER
    colors = [GROUP_COLORS[g] for g in group_order]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=300)
    axes = axes.reshape(-1)

    def grouped_bars(
        ax, categories, data_by_group, ylabel, xlabel, panel_label, ylim=(0, 100)
    ):
        x = np.arange(len(categories), dtype=float)
        width = 0.18
        offsets = (np.arange(len(group_order)) - (len(group_order) - 1) / 2.0) * width
        for gi, g in enumerate(group_order):
            ax.bar(
                x + offsets[gi],
                data_by_group[g],
                width=width,
                color=colors[gi],
                edgecolor="black",
                linewidth=0.8,
                label=g,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in categories], fontsize=TICK_FS)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=AXIS_LABEL_FS)
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=AXIS_LABEL_FS)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.18)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.text(
            0.03,
            0.88,
            panel_label,
            transform=ax.transAxes,
            fontsize=PANEL_FS,
            fontweight="bold",
        )
        ax.legend(frameon=True, fontsize=LEGEND_FS, loc="upper center")

    # A. Initial discriminations: exact categories 0, 1, 2, 3
    cats_A = [0, 1, 2, 3]
    data_A = {
        g: _percent_distribution(rows, "initial_discriminations", g, cats_A)
        for g in group_order
    }
    grouped_bars(
        axes[0],
        cats_A,
        data_A,
        "% of agents",
        "Initial discriminations achieved",
        "A",
    )

    # B. Total reversals: bins matching Waltz-style labels.
    cats_B = ["0", "1 or 2", "3 or 4", "5 or 6"]

    def total_rev_bin(v):
        v = int(round(float(v)))
        if v <= 0:
            return "0"
        if v <= 2:
            return "1 or 2"
        if v <= 4:
            return "3 or 4"
        return "5 or 6"

    data_B = {
        g: _percent_distribution(rows, "total_reversals", g, cats_B, total_rev_bin)
        for g in group_order
    }
    grouped_bars(
        axes[1],
        cats_B,
        data_B,
        "% of agents",
        "Total reversals achieved",
        "B",
    )

    # C. First reversals: exact categories 0, 1, 2, 3
    cats_C = [0, 1, 2, 3]
    data_C = {
        g: _percent_distribution(rows, "first_reversals", g, cats_C)
        for g in group_order
    }
    grouped_bars(
        axes[2],
        cats_C,
        data_C,
        "% of agents",
        "First reversals achieved",
        "C",
    )

    # D. Error rates: mean +/- SEM in percent for discrimination and first reversal.
    ax = axes[3]
    phases = ["Initial discrimination", "First reversal"]
    metrics = ["discrimination_error_rate", "first_reversal_error_rate"]
    x = np.arange(len(phases), dtype=float)
    width = 0.18
    offsets = (np.arange(len(group_order)) - (len(group_order) - 1) / 2.0) * width
    for gi, g in enumerate(group_order):
        means, sems = [], []
        for m in metrics:
            vals = get_group_values(rows, g, m) * 100.0
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))
        ax.bar(
            x + offsets[gi],
            means,
            width=width,
            yerr=sems,
            capsize=3,
            color=colors[gi],
            edgecolor="black",
            linewidth=0.8,
            label=g,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=TICK_FS)
    ax.set_ylabel("% of error trials", fontweight="bold")
    ax.set_xlabel("Task phase", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.18)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.text(
        0.03, 0.88, "D", transform=ax.transAxes, fontsize=PANEL_FS, fontweight="bold"
    )
    ax.legend(frameon=True, fontsize=LEGEND_FS, loc="upper center")

    fig.suptitle(
        "Waltz-like reversal-learning performance", fontweight="bold", fontsize=TITLE_FS
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def simulate_one_group(
    group, n_agents, base_seed, dp, mp, task, nmda, save_histories, out_dir
):
    rows, histories = [], [] if save_histories else None
    for agent_idx in range(n_agents):
        seed0 = base_seed + 100000 * agent_idx
        rng_spine = np.random.default_rng(seed0 + 1)
        rng_seg = np.random.default_rng(seed0 + 2)
        rng_map = np.random.default_rng(seed0 + 3)
        rng_sim = np.random.default_rng(seed0 + 4)
        spine = generate_spines(rng_spine, group.spine_params, group.spine_group)
        seg_id = kmeans_segments(spine["pos"], dp.n_segments, dp.kmeans_iters, rng_seg)
        active_k = max(
            1, int(round(group.spine_params.n_spines * mp.active_frac_each_option))
        )
        pair_maps = make_pair_option_spine_sets(
            rng_map,
            group.mapping_mode,
            task.n_pairs,
            active_k,
            spine["xl_mask"],
            spine["small_mask"],
            mp,
        )
        res = simulate_waltz_agent(
            rng_sim, spine, seg_id, pair_maps, task, nmda, group.learning_params
        )
        row = {
            "group_name": group.group_name,
            "agent_idx": agent_idx,
            "seed": seed0,
            "spine_factor": group.spine_factor,
            "da_factor": group.da_factor,
            **res["summary"],
            "xl_fraction": float(np.mean(spine["xl_mask"])),
            "small_fraction": float(np.mean(spine["small_mask"])),
            "mean_spine_size": float(np.mean(spine["weights"])),
            "sd_spine_size": float(np.std(spine["weights"], ddof=1)),
            "max_spine_size": float(np.max(spine["weights"])),
        }
        rows.append(row)
        if histories is not None:
            histories.append(
                {
                    "group_name": group.group_name,
                    "agent_idx": agent_idx,
                    "seed": seed0,
                    "summary": res["summary"],
                    "pair_input_rows": res["pair_input_rows"],
                    "trial_rows": res["trial_rows"],
                }
            )
    if histories is not None:
        group_dir = os.path.join(out_dir, group.group_name.lower())
        ensure_dir(group_dir)
        with open(
            os.path.join(group_dir, "agent_histories.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)
    return rows, histories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out_waltz_spine_weighted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_agents", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--save_histories", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--n_spines", type=int, default=4000)
    ap.add_argument("--lognorm_mu", type=float, default=-0.5)
    ap.add_argument("--lognorm_sigma", type=float, default=0.45)
    ap.add_argument("--xl_size_threshold", type=float, default=0.8)
    ap.add_argument("--small_loss_threshold", type=float, default=0.25)
    ap.add_argument("--small_loss_rate", type=float, default=0.25)
    ap.add_argument("--xl_extra_mu", type=float, default=0.0)
    ap.add_argument("--xl_extra_sigma", type=float, default=0.22)
    ap.add_argument("--xl_cluster_std_hc", type=float, default=18.0)
    ap.add_argument("--xl_cluster_std_sz", type=float, default=8.0)
    ap.add_argument("--active_frac_each_option", type=float, default=0.05)
    ap.add_argument("--healthy_xl_bias_A", type=float, default=0.50)
    ap.add_argument("--healthy_xl_bias_B", type=float, default=0.50)
    ap.add_argument("--pathological_xl_bias_A", type=float, default=0.80)
    ap.add_argument("--pathological_small_bias_B", type=float, default=0.80)
    ap.add_argument("--randomize_pathological_xl_option", action="store_true")
    ap.add_argument(
        "--input_mode", type=str, default="total", choices=["linear", "nmda", "total"]
    )
    ap.add_argument("--no_nmda", action="store_true")
    ap.add_argument("--nmda_gain", type=float, default=3.5)
    ap.add_argument("--theta_quantile", type=float, default=0.92)
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--alpha_neg_scale_da", type=float, default=0.35)
    ap.add_argument("--init_weight", type=float, default=0.0)
    ap.add_argument("--prediction_bias", type=float, default=0.0)
    ap.add_argument("--weight_clip", type=float, default=5.0)
    ap.add_argument("--beta_choice", type=float, default=5.0)
    ap.add_argument("--choice_value_noise", type=float, default=0.0)
    ap.add_argument("--n_pairs", type=int, default=3)
    ap.add_argument("--n_reversals_max_per_pair", type=int, default=2)
    ap.add_argument("--max_trials_per_stage", type=int, default=50)
    ap.add_argument("--criterion_window", type=int, default=10)
    ap.add_argument("--criterion_correct", type=int, default=9)
    ap.add_argument("--p_correct_reward", type=float, default=0.80)
    ap.add_argument("--p_incorrect_reward", type=float, default=0.20)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    dp = DendriteParams()
    mp = MappingParams(
        args.active_frac_each_option,
        args.healthy_xl_bias_A,
        args.healthy_xl_bias_B,
        args.pathological_xl_bias_A,
        args.pathological_small_bias_B,
        args.randomize_pathological_xl_option,
    )
    task = WaltzTaskParams(
        args.n_pairs,
        args.n_reversals_max_per_pair,
        args.max_trials_per_stage,
        args.criterion_window,
        args.criterion_correct,
        args.p_correct_reward,
        args.p_incorrect_reward,
    )
    nmda = NMDAGateParams(
        not args.no_nmda, args.input_mode, args.theta_quantile, args.nmda_gain
    )
    sp_hc = SpineParams(
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
    sp_sz = SpineParams(
        n_spines=args.n_spines,
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        small_loss_rate=args.small_loss_rate,
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
        xl_cluster_std=args.xl_cluster_std_sz,
    )
    lp_hc = LearningParams(
        args.alpha,
        1.0,
        args.prediction_bias,
        args.init_weight,
        args.weight_clip,
        args.beta_choice,
        args.choice_value_noise,
    )
    lp_da = LearningParams(
        args.alpha,
        args.alpha_neg_scale_da,
        args.prediction_bias,
        args.init_weight,
        args.weight_clip,
        args.beta_choice,
        args.choice_value_noise,
    )
    groups = [
        GroupConfig("HC", "healthy", "healthy_like", 0, 0, sp_hc, lp_hc),
        GroupConfig("Spine", "sz", "pathological_like", 1, 0, sp_sz, lp_hc),
        GroupConfig("DA", "healthy", "healthy_like", 0, 1, sp_hc, lp_da),
        GroupConfig("SZ", "sz", "pathological_like", 1, 1, sp_sz, lp_da),
    ]
    all_rows = []
    console_metrics = [
        "initial_discriminations",
        "first_reversals",
        "total_reversals",
        "discrimination_error_rate",
        "first_reversal_error_rate",
        "perseverative_error_rate_firstrev",
        "mean_option_input_imbalance",
        "mean_fired_xl_fraction",
    ]
    for gi, group in enumerate(groups):
        print(f"\n[{group.group_name}] running {args.n_agents} agents...")
        group_rows, _ = simulate_one_group(
            group,
            args.n_agents,
            args.seed + 10000000 * (gi + 1),
            dp,
            mp,
            task,
            nmda,
            args.save_histories,
            args.out_dir,
        )
        all_rows.extend(group_rows)
        for metric in console_metrics:
            vals = np.array([float(r[metric]) for r in group_rows], dtype=float)
            print(f"  {metric}: mean={safe_mean(vals):.4f}, sd={safe_std(vals):.4f}")
    main_metrics = [
        "initial_discriminations",
        "first_reversals",
        "total_reversals",
        "discrimination_error_rate",
        "first_reversal_error_rate",
        "perseverative_error_rate_firstrev",
        "n_trials_total",
        "mean_abs_pe",
        "mean_delta",
        "mean_x_A",
        "mean_x_B",
        "mean_option_input_imbalance",
        "mean_option_input_ratio",
        "mean_fired_xl_fraction",
        "mean_A_xl_fraction",
        "mean_B_xl_fraction",
        "mean_input_A_total",
        "mean_input_B_total",
        "xl_fraction",
        "small_fraction",
        "mean_spine_size",
        "sd_spine_size",
        "max_spine_size",
    ]
    rng_stats = np.random.default_rng(args.seed + 999)
    summary_rows = group_summary(all_rows, main_metrics, args.n_boot, rng_stats)
    anova = {
        m: factorial_anova_one_metric(
            all_rows, m, args.n_perm, np.random.default_rng(args.seed + 777)
        )
        for m in main_metrics
    }
    write_csv(os.path.join(args.out_dir, "agent_metrics.csv"), all_rows)
    write_csv(os.path.join(args.out_dir, "group_summary.csv"), summary_rows)
    write_csv(
        os.path.join(args.out_dir, "factorial_anova.csv"), flatten_anova_rows(anova)
    )
    with open(
        os.path.join(args.out_dir, "summary_all.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "args": vars(args),
                "group_summary": summary_rows,
                "factorial_anova": anova,
                "agent_metrics": all_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[Saved] {os.path.join(args.out_dir, 'agent_metrics.csv')}")
    print(f"[Saved] {os.path.join(args.out_dir, 'group_summary.csv')}")
    print(f"[Saved] {os.path.join(args.out_dir, 'factorial_anova.csv')}")
    print(f"[Saved] {os.path.join(args.out_dir, 'summary_all.json')}")
    if not args.no_plot:
        plot_metrics = [
            "initial_discriminations",
            "first_reversals",
            "total_reversals",
            "first_reversal_error_rate",
            "perseverative_error_rate_firstrev",
            "mean_fired_xl_fraction",
            "mean_option_input_imbalance",
            "mean_option_input_ratio",
        ]
        plot_path = os.path.join(args.out_dir, "summary_spine_weighted_waltz.png")
        plot_summary_metrics(all_rows, plot_metrics, plot_path)
        print(f"[Saved] {plot_path}")

        waltz_fig_path = os.path.join(args.out_dir, "summary_waltz_reference_style.png")
        plot_waltz_reference_style(all_rows, waltz_fig_path)
        print(f"[Saved] {waltz_fig_path}")
    print("\n=== 2x2 factorial analysis: compact summary ===")
    for metric in console_metrics:
        print(f"\n[{metric}]")
        for effect_name, vals in anova[metric].items():
            print(
                f"  {effect_name}: F={vals['F']:.4f}, p_perm={vals['p_permutation']:.4f}, p_param={vals['p_parametric']:.4f}"
            )


if __name__ == "__main__":
    main()
