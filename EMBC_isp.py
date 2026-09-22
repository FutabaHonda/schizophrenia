#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_isp_spine_weighted_learning.py

ISP-like task simulation using a spine-weighted associative learning model.

Purpose
-------
This script is designed for the hypothesis:
    spine size distribution -> stimulus input sum -> prediction -> PE learning -> RT

It does NOT attempt to reproduce the computational model of Katthagen et al.
Instead, it attempts to reproduce the behavioral RT pattern of an implicit
salience paradigm using a mechanistic link from spine-size distribution to
stimulus-specific input strength.

Core assumptions
----------------
1. Each stimulus feature manifestation activates a subset of dendritic spines.
2. Spine size is treated as synaptic input weight.
3. A pathological spine distribution contains fewer small spines and more
   extra-large (XL) spines, following a lognormal-size distribution idea.
4. Stimuli that happen to activate XL-rich subsets obtain larger input sums.
5. Prediction is computed from learned feature weights multiplied by the
   spine-derived input strength.
6. Delta-rule learning is scaled by the same input strength, so an XL-rich
   irrelevant feature can acquire value.
7. DA abnormality is implemented as reduced learning from negative prediction
   errors, making wrongly acquired irrelevant value harder to decrease.

Outputs
-------
- agent_metrics.csv
- group_summary.csv
- factorial_anova.csv
- summary_all.json
- summary_spine_weighted_isp.png

Groups
------
HC    = healthy spine distribution + normal negative learning
Spine = pathological spine distribution + normal negative learning
DA    = healthy spine distribution + reduced negative learning
SZ    = pathological spine distribution + reduced negative learning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


# =========================================================
# Utils
# =========================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sigmoid(x: float) -> float:
    # numerically stable enough for this scale
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1))


def safe_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    ci: float = 95.0,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])

    n = values.size
    boot_means = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(values, size=n, replace=True))

    a = (100.0 - ci) / 2.0
    return float(np.percentile(boot_means, a)), float(
        np.percentile(boot_means, 100.0 - a)
    )


# =========================================================
# Parameters
# =========================================================


@dataclass
class SpineParams:
    n_spines: int = 4000
    feat_dim: int = 2
    feat_range: float = 100.0

    # Lognormal spine-size distribution.
    lognorm_mu: float = -0.5
    lognorm_sigma: float = 0.45

    # XL and small definitions.
    xl_size_threshold: float = 0.8
    small_loss_threshold: float = 0.25

    # Pathological morphology.
    small_loss_rate: float = 0.25
    xl_extra_mu: float = 0.0
    xl_extra_sigma: float = 0.22

    # Spatial clustering of XL spines.
    xl_n_clusters: int = 12
    xl_cluster_std: float = 12.0


@dataclass
class DendriteParams:
    n_segments: int = 80
    kmeans_iters: int = 20


@dataclass
class ISPTaskParams:
    n_trials: int = 160
    block_len: int = 20
    dimension_switch_trial: int = 80
    p_relevant_high: float = 0.80
    p_relevant_low: float = 0.20


@dataclass
class FeatureMappingParams:
    active_frac_each_feature: float = 0.015

    # Healthy condition: no intentional XL/small assignment.
    healthy_mode: str = "random"

    # Pathological condition: one manifestation in each feature dimension is XL-rich,
    # the other is small-rich. This creates input asymmetry within irrelevant features.
    pathological_mode: str = "xl_vs_small"
    xl_bias_pathological: float = 0.80
    small_bias_pathological: float = 0.80


@dataclass
class NMDAGateParams:
    # If use_nmda is true, segment inputs above this quantile are amplified.
    theta_quantile: float = 0.92
    nmda_gain: float = 3.5
    use_nmda: bool = True


@dataclass
class LearningParams:
    alpha: float = 0.04
    alpha_neg_scale: float = 1.0
    init_weight: float = 0.0
    bias: float = 0.0
    weight_clip: float = 5.0


@dataclass
class RTParams:
    beta0: float = 6.18
    beta_abs_pe: float = 0.35
    beta_circle: float = 0.05
    beta_irrel_salience: float = 0.20
    beta_irrel_input: float = 0.03
    rt_noise: float = 0.02


@dataclass
class GroupConfig:
    group_name: str
    spine_group: str
    spine_factor: int
    da_factor: int
    spine_params: SpineParams
    learning_params: LearningParams
    mapping_mode: str


# =========================================================
# Spine generation
# =========================================================


def generate_spines_paper_like(
    rng: np.random.Generator,
    sp: SpineParams,
    group: str,
) -> Dict[str, np.ndarray]:
    """Generate spine sizes and positions.

    group='healthy': lognormal sizes.
    group='sz': remove a fraction of small spines and replace them with larger
                lognormal spines, then cluster XL spine positions more tightly.
    """
    n = sp.n_spines
    dim = sp.feat_dim

    base_sizes = rng.lognormal(sp.lognorm_mu, sp.lognorm_sigma, size=n)

    if group == "healthy":
        sizes = base_sizes.copy()
    elif group == "sz":
        sizes = base_sizes.copy()
        small_idx = np.where(sizes < sp.small_loss_threshold)[0]
        keep = np.ones(n, dtype=bool)
        n_drop = int(round(sp.small_loss_rate * small_idx.size))
        if n_drop > 0 and small_idx.size > 0:
            drop_idx = rng.choice(small_idx, size=n_drop, replace=False)
            keep[drop_idx] = False

        surviving = sizes[keep]
        n_missing = n - surviving.size
        if n_missing > 0:
            extra = rng.lognormal(sp.xl_extra_mu, sp.xl_extra_sigma, size=n_missing)
            sizes = np.concatenate([surviving, extra])
        else:
            sizes = surviving[:n].copy()
        rng.shuffle(sizes)
    else:
        raise ValueError("group must be 'healthy' or 'sz'")

    xl_mask = sizes >= sp.xl_size_threshold
    small_mask = sizes <= sp.small_loss_threshold
    weights = sizes.astype(float)

    # Positions: non-XL uniform, XL clustered.
    pos = np.empty((n, dim), dtype=float)
    non_idx = np.where(~xl_mask)[0]
    pos[non_idx] = rng.uniform(-sp.feat_range, sp.feat_range, size=(non_idx.size, dim))

    xl_idx = np.where(xl_mask)[0]
    if xl_idx.size > 0:
        centers = rng.uniform(
            -sp.feat_range, sp.feat_range, size=(sp.xl_n_clusters, dim)
        )
        rng.shuffle(xl_idx)
        chunks = np.array_split(xl_idx, sp.xl_n_clusters)
        for c, ids in enumerate(chunks):
            if ids.size == 0:
                continue
            pos[ids] = centers[c] + rng.normal(
                0.0, sp.xl_cluster_std, size=(ids.size, dim)
            )

    return {
        "sizes": sizes,
        "weights": weights,
        "pos": pos,
        "xl_mask": xl_mask,
        "small_mask": small_mask,
    }


# =========================================================
# Dendrite segmentation
# =========================================================


def kmeans_segments(
    pos: np.ndarray,
    n_segments: int,
    iters: int,
    rng: np.random.Generator,
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
            if idx.size == 0:
                centers[j] = pos[rng.integers(0, n)]
            else:
                centers[j] = pos[idx].mean(axis=0)
    return seg


# =========================================================
# Feature-to-spine mapping
# =========================================================


def sample_bool_mask(
    rng: np.random.Generator,
    n_total: int,
    active_k: int,
    preferred: np.ndarray | None = None,
    other: np.ndarray | None = None,
    pref_frac: float = 0.0,
) -> np.ndarray:
    if preferred is None or other is None:
        idx = rng.choice(np.arange(n_total), size=active_k, replace=False)
    else:
        pref_frac = float(np.clip(pref_frac, 0.0, 1.0))
        k_pref = int(round(active_k * pref_frac))
        k_other = active_k - k_pref
        chosen = []
        if k_pref > 0 and preferred.size > 0:
            chosen.append(
                rng.choice(preferred, size=min(k_pref, preferred.size), replace=False)
            )
        if k_other > 0 and other.size > 0:
            chosen.append(
                rng.choice(other, size=min(k_other, other.size), replace=False)
            )
        if chosen:
            idx = np.concatenate(chosen)
            if idx.size < active_k:
                rest = np.setdiff1d(np.arange(n_total), idx)
                fill = rng.choice(rest, size=active_k - idx.size, replace=False)
                idx = np.concatenate([idx, fill])
        else:
            idx = rng.choice(np.arange(n_total), size=active_k, replace=False)

    m = np.zeros(n_total, dtype=bool)
    m[idx] = True
    return m


def make_feature_spine_sets(
    rng: np.random.Generator,
    mapping_mode: str,
    active_k: int,
    xl_mask: np.ndarray,
    small_mask: np.ndarray,
    mp: FeatureMappingParams,
) -> Dict[str, np.ndarray]:
    """Create four feature manifestation masks.

    shape_0, shape_1, color_0, color_1

    Healthy: random subsets.
    Pathological default: within each dimension, manifestation 0 is XL-rich and
    manifestation 1 is small-rich. This creates an input asymmetry that can be
    irrelevant depending on the current task phase.
    """
    n = xl_mask.size
    all_ids = np.arange(n)
    xl_ids = np.where(xl_mask)[0]
    non_xl_ids = np.where(~xl_mask)[0]
    small_ids = np.where(small_mask)[0]
    non_small_ids = np.where(~small_mask)[0]

    if mapping_mode == "healthy_like":
        return {
            "shape_0": sample_bool_mask(rng, n, active_k),
            "shape_1": sample_bool_mask(rng, n, active_k),
            "color_0": sample_bool_mask(rng, n, active_k),
            "color_1": sample_bool_mask(rng, n, active_k),
        }

    if mapping_mode == "pathological_like":
        return {
            "shape_0": sample_bool_mask(
                rng, n, active_k, xl_ids, non_xl_ids, mp.xl_bias_pathological
            ),
            "shape_1": sample_bool_mask(
                rng, n, active_k, small_ids, non_small_ids, mp.small_bias_pathological
            ),
            "color_0": sample_bool_mask(
                rng, n, active_k, xl_ids, non_xl_ids, mp.xl_bias_pathological
            ),
            "color_1": sample_bool_mask(
                rng, n, active_k, small_ids, non_small_ids, mp.small_bias_pathological
            ),
        }

    if mapping_mode == "pathological_random":
        # Useful control: pathological spine distribution without intentional
        # feature-level XL/small assignment.
        return {
            "shape_0": sample_bool_mask(rng, n, active_k),
            "shape_1": sample_bool_mask(rng, n, active_k),
            "color_0": sample_bool_mask(rng, n, active_k),
            "color_1": sample_bool_mask(rng, n, active_k),
        }

    raise ValueError(f"Unknown mapping_mode: {mapping_mode}")


# =========================================================
# ISP task schedule
# =========================================================


def get_relevant_dimension(trial_idx: int, task: ISPTaskParams) -> str:
    return "color" if trial_idx < task.dimension_switch_trial else "shape"


def get_high_manifestation(trial_idx: int, task: ISPTaskParams) -> int:
    return int((trial_idx // task.block_len) % 2)


def get_reward_probability(
    trial_idx: int,
    shape_val: int,
    color_val: int,
    task: ISPTaskParams,
) -> float:
    rel_dim = get_relevant_dimension(trial_idx, task)
    high_val = get_high_manifestation(trial_idx, task)
    if rel_dim == "color":
        return task.p_relevant_high if color_val == high_val else task.p_relevant_low
    return task.p_relevant_high if shape_val == high_val else task.p_relevant_low


def sample_cue(rng: np.random.Generator) -> Tuple[int, int]:
    return int(rng.integers(0, 2)), int(rng.integers(0, 2))


# =========================================================
# Spine input computation
# =========================================================


def compute_feature_inputs(
    active_shape: np.ndarray,
    active_color: np.ndarray,
    weights: np.ndarray,
    seg_id: np.ndarray,
    nmda: NMDAGateParams,
    input_mode: str,
) -> Dict[str, float | np.ndarray]:
    active = active_shape | active_color
    n_segments = int(seg_id.max()) + 1
    active_idx = np.where(active)[0]

    seg_sums = np.zeros(n_segments, dtype=float)
    np.add.at(seg_sums, seg_id[active_idx], weights[active_idx])

    if nmda.use_nmda:
        theta = float(np.quantile(seg_sums, nmda.theta_quantile))
        fired_seg = seg_sums > theta
    else:
        theta = float("inf")
        fired_seg = np.zeros(n_segments, dtype=bool)

    fired_spine_mask = active & fired_seg[seg_id]

    amp = np.ones_like(weights, dtype=float)
    amp[fired_spine_mask] = nmda.nmda_gain

    lin_shape = float(np.sum(weights[active_shape]))
    lin_color = float(np.sum(weights[active_color]))

    total_shape = float(np.sum(weights[active_shape] * amp[active_shape]))
    total_color = float(np.sum(weights[active_color] * amp[active_color]))

    excess_amp = amp - 1.0
    nmda_shape = float(np.sum(weights[active_shape] * excess_amp[active_shape]))
    nmda_color = float(np.sum(weights[active_color] * excess_amp[active_color]))

    denom_shape = float(np.sum(active_shape) * (np.mean(weights) + 1e-12))
    denom_color = float(np.sum(active_color) * (np.mean(weights) + 1e-12))

    if input_mode == "linear":
        x_shape = lin_shape / denom_shape
        x_color = lin_color / denom_color
    elif input_mode == "nmda_excess":
        x_shape = nmda_shape / denom_shape
        x_color = nmda_color / denom_color
    elif input_mode == "total":
        x_shape = total_shape / denom_shape
        x_color = total_color / denom_color
    else:
        raise ValueError("input_mode must be linear, nmda_excess, or total")

    if np.any(fired_spine_mask):
        fired_xl_fraction = np.nan  # filled outside where xl_mask is known
    else:
        fired_xl_fraction = np.nan

    return {
        "x_shape": float(x_shape),
        "x_color": float(x_color),
        "lin_shape": lin_shape,
        "lin_color": lin_color,
        "total_shape": total_shape,
        "total_color": total_color,
        "nmda_shape": nmda_shape,
        "nmda_color": nmda_color,
        "theta": theta,
        "n_fired_segments": int(np.sum(fired_seg)),
        "fired_spine_mask": fired_spine_mask,
        "seg_sums": seg_sums,
    }


# =========================================================
# Metrics
# =========================================================


def compute_aberrant_salience_ms(
    rt_ms: np.ndarray,
    cue_hist: np.ndarray,
    rel_dim_hist: np.ndarray,
) -> Tuple[float, float, float]:
    # First half: color relevant, shape irrelevant.
    color_rel_idx = np.where(rel_dim_hist == "color")[0]
    shape_bias = float("nan")
    if color_rel_idx.size > 0:
        idx0 = color_rel_idx[cue_hist[color_rel_idx, 0] == 0]
        idx1 = color_rel_idx[cue_hist[color_rel_idx, 0] == 1]
        if idx0.size > 0 and idx1.size > 0:
            shape_bias = abs(safe_mean(rt_ms[idx0]) - safe_mean(rt_ms[idx1]))

    # Second half: shape relevant, color irrelevant.
    shape_rel_idx = np.where(rel_dim_hist == "shape")[0]
    color_bias = float("nan")
    if shape_rel_idx.size > 0:
        idx0 = shape_rel_idx[cue_hist[shape_rel_idx, 1] == 0]
        idx1 = shape_rel_idx[cue_hist[shape_rel_idx, 1] == 1]
        if idx0.size > 0 and idx1.size > 0:
            color_bias = abs(safe_mean(rt_ms[idx0]) - safe_mean(rt_ms[idx1]))

    total = safe_mean(np.asarray([shape_bias, color_bias], dtype=float))
    return total, shape_bias, color_bias


# =========================================================
# Core simulation
# =========================================================


def simulate_isp_agent(
    rng: np.random.Generator,
    spine: Dict[str, np.ndarray],
    seg_id: np.ndarray,
    feature_maps: Dict[str, np.ndarray],
    task: ISPTaskParams,
    nmda: NMDAGateParams,
    learn: LearningParams,
    rtp: RTParams,
    input_mode: str,
) -> Dict:
    weights = spine["weights"]
    xl_mask = spine["xl_mask"]

    # Learned association weights for outcome A=1 (coin).
    # These are not probabilities themselves; probabilities are generated by sigmoid.
    W_shape = np.array([learn.init_weight, learn.init_weight], dtype=float)
    W_color = np.array([learn.init_weight, learn.init_weight], dtype=float)

    n = task.n_trials
    cue_hist = np.zeros((n, 2), dtype=int)
    rel_dim_hist = np.empty(n, dtype=object)
    outcome_hist = np.zeros(n, dtype=int)
    expected_hist = np.zeros(n, dtype=int)

    log_rt_hist = np.zeros(n, dtype=float)
    rt_ms_hist = np.zeros(n, dtype=float)
    E_hist = np.zeros(n, dtype=float)
    delta_hist = np.zeros(n, dtype=float)

    W_shape_hist = np.zeros((n, 2), dtype=float)
    W_color_hist = np.zeros((n, 2), dtype=float)

    x_shape_hist = np.zeros(n, dtype=float)
    x_color_hist = np.zeros(n, dtype=float)
    lin_shape_hist = np.zeros(n, dtype=float)
    lin_color_hist = np.zeros(n, dtype=float)
    total_shape_hist = np.zeros(n, dtype=float)
    total_color_hist = np.zeros(n, dtype=float)
    nmda_shape_hist = np.zeros(n, dtype=float)
    nmda_color_hist = np.zeros(n, dtype=float)

    irrel_input_hist = np.zeros(n, dtype=float)
    irrel_weight_hist = np.zeros(n, dtype=float)
    irrel_salience_hist = np.zeros(n, dtype=float)
    rel_input_hist = np.zeros(n, dtype=float)
    rel_weight_hist = np.zeros(n, dtype=float)
    rel_salience_hist = np.zeros(n, dtype=float)

    fired_segment_count_hist = np.zeros(n, dtype=float)
    fired_xl_fraction_hist = np.zeros(n, dtype=float)

    for t in range(n):
        shape_val, color_val = sample_cue(rng)
        cue_hist[t] = [shape_val, color_val]

        rel_dim = get_relevant_dimension(t, task)
        rel_dim_hist[t] = rel_dim

        active_shape = feature_maps[f"shape_{shape_val}"]
        active_color = feature_maps[f"color_{color_val}"]

        inp = compute_feature_inputs(
            active_shape=active_shape,
            active_color=active_color,
            weights=weights,
            seg_id=seg_id,
            nmda=nmda,
            input_mode=input_mode,
        )
        x_shape = float(inp["x_shape"])
        x_color = float(inp["x_color"])

        x_shape_hist[t] = x_shape
        x_color_hist[t] = x_color
        lin_shape_hist[t] = float(inp["lin_shape"])
        lin_color_hist[t] = float(inp["lin_color"])
        total_shape_hist[t] = float(inp["total_shape"])
        total_color_hist[t] = float(inp["total_color"])
        nmda_shape_hist[t] = float(inp["nmda_shape"])
        nmda_color_hist[t] = float(inp["nmda_color"])
        fired_segment_count_hist[t] = float(inp["n_fired_segments"])

        fired_mask = inp["fired_spine_mask"]
        if np.any(fired_mask):
            fired_xl_fraction_hist[t] = float(np.mean(xl_mask[fired_mask]))
        else:
            fired_xl_fraction_hist[t] = 0.0

        # Spine-weighted prediction.
        z = learn.bias + W_shape[shape_val] * x_shape + W_color[color_val] * x_color
        E = sigmoid(z)
        E_hist[t] = E

        # Generate outcome from true task structure.
        p_coin = get_reward_probability(t, shape_val, color_val, task)
        A = 1 if rng.random() < p_coin else 0
        outcome_hist[t] = A

        high_val = get_high_manifestation(t, task)
        predicted_coin = (
            (color_val == high_val) if rel_dim == "color" else (shape_val == high_val)
        )
        expected_hist[t] = int(
            (predicted_coin and A == 1) or ((not predicted_coin) and A == 0)
        )

        # Global PE for the whole stimulus compound.
        delta = float(A - E)
        delta_hist[t] = delta

        alpha_eff = learn.alpha if delta >= 0.0 else learn.alpha * learn.alpha_neg_scale

        # Input-scaled associative learning.
        W_shape[shape_val] += alpha_eff * delta * x_shape
        W_color[color_val] += alpha_eff * delta * x_color
        W_shape = np.clip(W_shape, -learn.weight_clip, learn.weight_clip)
        W_color = np.clip(W_color, -learn.weight_clip, learn.weight_clip)

        W_shape_hist[t] = W_shape.copy()
        W_color_hist[t] = W_color.copy()

        if rel_dim == "color":
            irrel_x = x_shape
            irrel_w = W_shape[shape_val]
            rel_x = x_color
            rel_w = W_color[color_val]
        else:
            irrel_x = x_color
            irrel_w = W_color[color_val]
            rel_x = x_shape
            rel_w = W_shape[shape_val]

        irrel_input = float(irrel_x)
        irrel_weight = float(irrel_w)
        irrel_salience = float(abs(irrel_w * irrel_x))
        rel_input = float(rel_x)
        rel_weight = float(rel_w)
        rel_salience = float(abs(rel_w * rel_x))

        irrel_input_hist[t] = irrel_input
        irrel_weight_hist[t] = irrel_weight
        irrel_salience_hist[t] = irrel_salience
        rel_input_hist[t] = rel_input
        rel_weight_hist[t] = rel_weight
        rel_salience_hist[t] = rel_salience

        # RT proxy: behavioral expression of learned irrelevant salience.
        log_rt = (
            rtp.beta0
            + rtp.beta_abs_pe * abs(delta)
            + rtp.beta_circle * (1.0 if A == 0 else 0.0)
            + rtp.beta_irrel_salience * irrel_salience
            + rtp.beta_irrel_input * irrel_input
            + rng.normal(0.0, rtp.rt_noise)
        )
        log_rt_hist[t] = float(log_rt)
        rt_ms_hist[t] = float(np.exp(log_rt))

    as_ms, as_shape_ms, as_color_ms = compute_aberrant_salience_ms(
        rt_ms=rt_ms_hist,
        cue_hist=cue_hist,
        rel_dim_hist=rel_dim_hist,
    )

    expected_mask = expected_hist == 1
    unexpected_mask = expected_hist == 0
    coin_mask = outcome_hist == 1
    circle_mask = outcome_hist == 0

    summary = {
        # Katthagen-style behavioral summaries.
        "kat_aberrant_salience_ms": float(as_ms),
        "kat_shape_irrel_bias_ms": float(as_shape_ms),
        "kat_color_irrel_bias_ms": float(as_color_ms),
        "kat_expected_logrt": safe_mean(log_rt_hist[expected_mask]),
        "kat_unexpected_logrt": safe_mean(log_rt_hist[unexpected_mask]),
        "kat_unexpected_minus_expected": safe_mean(log_rt_hist[unexpected_mask])
        - safe_mean(log_rt_hist[expected_mask]),
        "kat_expected_rt_ms": safe_mean(rt_ms_hist[expected_mask]),
        "kat_unexpected_rt_ms": safe_mean(rt_ms_hist[unexpected_mask]),
        "kat_unexpected_minus_expected_ms": safe_mean(rt_ms_hist[unexpected_mask])
        - safe_mean(rt_ms_hist[expected_mask]),
        "kat_coin_logrt": safe_mean(log_rt_hist[coin_mask]),
        "kat_circle_logrt": safe_mean(log_rt_hist[circle_mask]),
        "kat_circle_minus_coin": safe_mean(log_rt_hist[circle_mask])
        - safe_mean(log_rt_hist[coin_mask]),
        "kat_coin_rt_ms": safe_mean(rt_ms_hist[coin_mask]),
        "kat_circle_rt_ms": safe_mean(rt_ms_hist[circle_mask]),
        "kat_circle_minus_coin_ms": safe_mean(rt_ms_hist[circle_mask])
        - safe_mean(rt_ms_hist[coin_mask]),
        # Mechanistic summaries.
        "mean_abs_pe": float(np.mean(np.abs(delta_hist))),
        "mean_E": float(np.mean(E_hist)),
        "mean_x_shape": float(np.mean(x_shape_hist)),
        "mean_x_color": float(np.mean(x_color_hist)),
        "mean_irrel_input": float(np.mean(irrel_input_hist)),
        "mean_rel_input": float(np.mean(rel_input_hist)),
        "mean_irrel_weight": float(np.mean(irrel_weight_hist)),
        "mean_rel_weight": float(np.mean(rel_weight_hist)),
        "mean_abs_irrel_weight": float(np.mean(np.abs(irrel_weight_hist))),
        "mean_abs_rel_weight": float(np.mean(np.abs(rel_weight_hist))),
        "mean_irrel_salience": float(np.mean(irrel_salience_hist)),
        "mean_rel_salience": float(np.mean(rel_salience_hist)),
        "irrel_minus_rel_salience": float(
            np.mean(irrel_salience_hist) - np.mean(rel_salience_hist)
        ),
        "mean_fired_segment_count": float(np.mean(fired_segment_count_hist)),
        "mean_fired_xl_fraction": float(np.mean(fired_xl_fraction_hist)),
        "mean_nmda_shape": float(np.mean(nmda_shape_hist)),
        "mean_nmda_color": float(np.mean(nmda_color_hist)),
    }

    spine_stats = {
        "xl_fraction": float(np.mean(spine["xl_mask"])),
        "small_fraction": float(np.mean(spine["small_mask"])),
        "mean_spine_size": float(np.mean(spine["sizes"])),
        "sd_spine_size": float(np.std(spine["sizes"])),
        "max_spine_size": float(np.max(spine["sizes"])),
    }

    hist = {
        "cue_hist": cue_hist.tolist(),
        "rel_dim_hist": rel_dim_hist.tolist(),
        "outcome_hist": outcome_hist.tolist(),
        "expected_hist": expected_hist.tolist(),
        "log_rt_hist": log_rt_hist.tolist(),
        "rt_ms_hist": rt_ms_hist.tolist(),
        "E_hist": E_hist.tolist(),
        "delta_hist": delta_hist.tolist(),
        "W_shape_hist": W_shape_hist.tolist(),
        "W_color_hist": W_color_hist.tolist(),
        "x_shape_hist": x_shape_hist.tolist(),
        "x_color_hist": x_color_hist.tolist(),
        "irrel_input_hist": irrel_input_hist.tolist(),
        "irrel_weight_hist": irrel_weight_hist.tolist(),
        "irrel_salience_hist": irrel_salience_hist.tolist(),
        "rel_input_hist": rel_input_hist.tolist(),
        "rel_weight_hist": rel_weight_hist.tolist(),
        "rel_salience_hist": rel_salience_hist.tolist(),
        "fired_segment_count_hist": fired_segment_count_hist.tolist(),
        "fired_xl_fraction_hist": fired_xl_fraction_hist.tolist(),
    }

    return {"summary": summary, "spine_stats": spine_stats, "hist": hist}


# =========================================================
# Statistical analysis
# =========================================================


def fit_ols_sse(y: np.ndarray, X: np.ndarray) -> Tuple[float, int]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok]
    if y.size == 0:
        return float("nan"), 0
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(np.sum(resid**2))
    rank = np.linalg.matrix_rank(X)
    return sse, int(y.size - rank)


def factorial_anova_one_metric(
    rows: List[Dict],
    metric: str,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    y = np.array([float(r[metric]) for r in rows], dtype=float)
    spine = np.array([int(r["spine_factor"]) for r in rows], dtype=float)
    da = np.array([int(r["da_factor"]) for r in rows], dtype=float)
    inter = spine * da

    X_full = np.column_stack([np.ones_like(spine), spine, da, inter])
    full_sse, full_df = fit_ols_sse(y, X_full)

    effects = {
        "Spine": [0, 2, 3],
        "DA": [0, 1, 3],
        "Spine_x_DA": [0, 1, 2],
    }

    out: Dict[str, Dict[str, float]] = {}
    for name, keep_cols in effects.items():
        X_red = X_full[:, keep_cols]
        red_sse, red_df = fit_ols_sse(y, X_red)
        df_num = red_df - full_df
        df_den = full_df

        if (
            df_num <= 0
            or df_den <= 0
            or not np.isfinite(red_sse)
            or not np.isfinite(full_sse)
        ):
            f_obs = float("nan")
        else:
            ms_num = (red_sse - full_sse) / df_num
            ms_den = full_sse / df_den
            f_obs = float(ms_num / ms_den) if ms_den > 0 else float("nan")

        if HAS_SCIPY and np.isfinite(f_obs):
            p_param = float(stats.f.sf(f_obs, df_num, df_den))
        else:
            p_param = float("nan")

        p_perm = float("nan")
        if n_perm > 0 and np.isfinite(f_obs):
            count = 0
            used = 0
            for _ in range(n_perm):
                yp = rng.permutation(y)
                fsse, fdf = fit_ols_sse(yp, X_full)
                rsse, rdf = fit_ols_sse(yp, X_red)
                p_df_num = rdf - fdf
                p_df_den = fdf
                if p_df_num <= 0 or p_df_den <= 0:
                    continue
                p_ms_num = (rsse - fsse) / p_df_num
                p_ms_den = fsse / p_df_den
                if p_ms_den <= 0:
                    continue
                f_perm = p_ms_num / p_ms_den
                used += 1
                if f_perm >= f_obs:
                    count += 1
            if used > 0:
                p_perm = float((count + 1) / (used + 1))

        out[name] = {
            "F": f_obs,
            "df_num": float(df_num),
            "df_den": float(df_den),
            "p_parametric": p_param,
            "p_permutation": p_perm,
        }

    return out


def group_summary(
    rows: List[Dict],
    metrics: List[str],
    n_boot: int,
    rng: np.random.Generator,
) -> List[Dict]:
    out = []
    for group in ["HC", "Spine", "DA", "SZ"]:
        group_rows = [r for r in rows if r["group_name"] == group]
        if not group_rows:
            continue
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in group_rows], dtype=float)
            lo, hi = bootstrap_ci(vals, rng=rng, n_boot=n_boot)
            out.append(
                {
                    "group_name": group,
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


# =========================================================
# IO and plotting
# =========================================================


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def flatten_anova(anova: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict]:
    out = []
    for metric, effects in anova.items():
        for effect, vals in effects.items():
            out.append({"metric": metric, "effect": effect, **vals})
    return out


# Large-font settings for A0 poster figures.
TITLE_FS = 24
AXIS_LABEL_FS = 19
TICK_FS = 16
LEGEND_FS = 15


def plot_summary_metrics(rows: List[Dict], metrics: List[str], out_path: str) -> None:
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available. Skipping plot.")
        return

    group_order = ["HC", "Spine", "DA", "SZ"]
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5.0 * n_rows), dpi=300)
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(123)

    for ax, metric in zip(axes, metrics):
        means = []
        sems = []
        for g in group_order:
            vals = np.array(
                [float(r[metric]) for r in rows if r["group_name"] == g], dtype=float
            )
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))
        x = np.arange(len(group_order))
        ax.bar(
            x, means, yerr=sems, capsize=4, edgecolor="black", linewidth=0.8, alpha=0.75
        )
        for i, g in enumerate(group_order):
            vals = np.array(
                [float(r[metric]) for r in rows if r["group_name"] == g], dtype=float
            )
            vals = vals[np.isfinite(vals)]
            jitter = rng.normal(0.0, 0.055, size=vals.size)
            ax.scatter(
                np.full(vals.size, i) + jitter, vals, s=12, alpha=0.45, color="black"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, fontsize=TICK_FS)
        ax.set_title(metric, fontweight="bold", fontsize=AXIS_LABEL_FS)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_FS)

    for i in range(len(metrics), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Spine-weighted ISP simulation", fontweight="bold", fontsize=TITLE_FS)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_katthagen_bar3(rows: List[Dict], out_path: str) -> None:
    """Create three side-by-side bar graphs for poster-style ISP summaries.

    Panels:
      1. Unexpected - Expected RT (ms)
      2. Circle - Coin RT (ms)
      3. Aberrant salience score (ms)

    Fixed colors requested by the user:
      HC blue, Spine yellow, DA orange, SZ red.
    """
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available. Skipping plot.")
        return

    group_order = ["HC", "Spine", "DA", "SZ"]
    group_colors = {
        "HC": "#1f77b4",  # blue
        "Spine": "#ffd21f",  # yellow
        "DA": "#ff7f0e",  # orange
        "SZ": "#d62728",  # red
    }
    panels = [
        (
            "kat_unexpected_minus_expected_ms",
            "Unexpected - Expected RT",
            "RT difference (ms)",
        ),
        ("kat_circle_minus_coin_ms", "Circle - Coin RT", "RT difference (ms)"),
        ("kat_aberrant_salience_ms", "Aberrant salience score", "Score (ms)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    rng = np.random.default_rng(123)

    for ax, (metric, title, ylabel) in zip(axes, panels):
        means = []
        sems = []
        for g in group_order:
            vals = np.array(
                [float(r[metric]) for r in rows if r["group_name"] == g], dtype=float
            )
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))

        x = np.arange(len(group_order))
        colors = [group_colors[g] for g in group_order]
        ax.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.90,
        )

        # Overlay agent-level points for transparency.
        for i, g in enumerate(group_order):
            vals = np.array(
                [float(r[metric]) for r in rows if r["group_name"] == g], dtype=float
            )
            vals = vals[np.isfinite(vals)]
            jitter = rng.normal(0.0, 0.055, size=vals.size)
            ax.scatter(
                np.full(vals.size, i) + jitter,
                vals,
                s=10,
                alpha=0.35,
                color="black",
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0, color="black", linewidth=0.7, alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, fontsize=TICK_FS)
        ax.set_title(title, fontweight="bold", fontsize=AXIS_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_FS)

    fig.suptitle("ISP behavioral RT metrics", fontweight="bold", fontsize=TITLE_FS)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Runner
# =========================================================


def simulate_one_group(
    group: GroupConfig,
    n_agents: int,
    base_seed: int,
    dend: DendriteParams,
    task: ISPTaskParams,
    fmap_params: FeatureMappingParams,
    nmda: NMDAGateParams,
    rtp: RTParams,
    input_mode: str,
    save_histories: bool,
    out_dir: str,
) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    histories: List[Dict] = []

    for agent_idx in range(n_agents):
        seed0 = base_seed + 100000 * agent_idx
        rng_spine = np.random.default_rng(seed0 + 1)
        rng_seg = np.random.default_rng(seed0 + 2)
        rng_map = np.random.default_rng(seed0 + 3)
        rng_sim = np.random.default_rng(seed0 + 4)

        spine = generate_spines_paper_like(
            rng_spine, group.spine_params, group.spine_group
        )
        seg_id = kmeans_segments(
            spine["pos"], dend.n_segments, dend.kmeans_iters, rng_seg
        )

        active_k = max(
            1,
            int(
                round(
                    group.spine_params.n_spines * fmap_params.active_frac_each_feature
                )
            ),
        )
        feature_maps = make_feature_spine_sets(
            rng_map,
            group.mapping_mode,
            active_k,
            spine["xl_mask"],
            spine["small_mask"],
            fmap_params,
        )

        res = simulate_isp_agent(
            rng=rng_sim,
            spine=spine,
            seg_id=seg_id,
            feature_maps=feature_maps,
            task=task,
            nmda=nmda,
            learn=group.learning_params,
            rtp=rtp,
            input_mode=input_mode,
        )

        row = {
            "group_name": group.group_name,
            "agent_idx": agent_idx,
            "seed": seed0,
            "spine_factor": group.spine_factor,
            "da_factor": group.da_factor,
            **res["summary"],
            **res["spine_stats"],
        }
        rows.append(row)

        if save_histories:
            histories.append(
                {
                    "group_name": group.group_name,
                    "agent_idx": agent_idx,
                    "seed": seed0,
                    "summary": res["summary"],
                    "spine_stats": res["spine_stats"],
                    "hist": res["hist"],
                }
            )

    if save_histories:
        group_dir = os.path.join(out_dir, group.group_name.lower())
        ensure_dir(group_dir)
        with open(
            os.path.join(group_dir, "agent_histories.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)

    return rows, histories


def build_groups(args: argparse.Namespace) -> List[GroupConfig]:
    sp_hc = SpineParams(
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        small_loss_rate=0.0,
        xl_cluster_std=args.xl_cluster_std_hc,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
    )
    sp_sz = SpineParams(
        lognorm_mu=args.lognorm_mu,
        lognorm_sigma=args.lognorm_sigma,
        small_loss_rate=args.small_loss_rate,
        xl_cluster_std=args.xl_cluster_std_sz,
        xl_size_threshold=args.xl_size_threshold,
        small_loss_threshold=args.small_loss_threshold,
        xl_extra_mu=args.xl_extra_mu,
        xl_extra_sigma=args.xl_extra_sigma,
    )

    lp_hc = LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=1.0,
        init_weight=args.init_weight,
        bias=args.prediction_bias,
        weight_clip=args.weight_clip,
    )
    lp_da = LearningParams(
        alpha=args.alpha,
        alpha_neg_scale=args.alpha_neg_scale_da,
        init_weight=args.init_weight,
        bias=args.prediction_bias,
        weight_clip=args.weight_clip,
    )

    return [
        GroupConfig("HC", "healthy", 0, 0, sp_hc, lp_hc, "healthy_like"),
        GroupConfig("Spine", "sz", 1, 0, sp_sz, lp_hc, args.pathological_mapping_mode),
        GroupConfig("DA", "healthy", 0, 1, sp_hc, lp_da, "healthy_like"),
        GroupConfig("SZ", "sz", 1, 1, sp_sz, lp_da, args.pathological_mapping_mode),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out_isp_spine_weighted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_agents", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--save_histories", action="store_true")
    ap.add_argument("--no_plot", action="store_true")

    # Task and input model.
    ap.add_argument(
        "--input_mode",
        type=str,
        default="total",
        choices=["linear", "nmda_excess", "total"],
    )
    ap.add_argument("--use_nmda", action="store_true", default=True)
    ap.add_argument("--no_nmda", action="store_false", dest="use_nmda")
    ap.add_argument("--nmda_gain", type=float, default=3.5)
    ap.add_argument("--theta_quantile", type=float, default=0.92)

    # Spine distribution parameters.
    ap.add_argument("--lognorm_mu", type=float, default=-0.5)
    ap.add_argument("--lognorm_sigma", type=float, default=0.45)
    ap.add_argument("--xl_size_threshold", type=float, default=0.8)
    ap.add_argument("--small_loss_threshold", type=float, default=0.25)
    ap.add_argument("--small_loss_rate", type=float, default=0.25)
    ap.add_argument("--xl_extra_mu", type=float, default=0.0)
    ap.add_argument("--xl_extra_sigma", type=float, default=0.22)
    ap.add_argument("--xl_cluster_std_hc", type=float, default=18.0)
    ap.add_argument("--xl_cluster_std_sz", type=float, default=8.0)

    # Feature mapping.
    ap.add_argument("--active_frac_each_feature", type=float, default=0.015)
    ap.add_argument("--xl_bias_pathological", type=float, default=0.80)
    ap.add_argument("--small_bias_pathological", type=float, default=0.80)
    ap.add_argument(
        "--pathological_mapping_mode",
        type=str,
        default="pathological_like",
        choices=["pathological_like", "pathological_random"],
    )

    # Learning and DA.
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--alpha_neg_scale_da", type=float, default=0.35)
    ap.add_argument("--init_weight", type=float, default=0.0)
    ap.add_argument("--prediction_bias", type=float, default=0.0)
    ap.add_argument("--weight_clip", type=float, default=5.0)

    # RT model.
    ap.add_argument("--beta0", type=float, default=6.18)
    ap.add_argument("--beta_abs_pe", type=float, default=0.35)
    ap.add_argument("--beta_circle", type=float, default=0.05)
    ap.add_argument("--beta_irrel_salience", type=float, default=0.20)
    ap.add_argument("--beta_irrel_input", type=float, default=0.03)
    ap.add_argument("--rt_noise", type=float, default=0.02)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    dend = DendriteParams()
    task = ISPTaskParams()
    fmap_params = FeatureMappingParams(
        active_frac_each_feature=args.active_frac_each_feature,
        xl_bias_pathological=args.xl_bias_pathological,
        small_bias_pathological=args.small_bias_pathological,
    )
    nmda = NMDAGateParams(
        theta_quantile=args.theta_quantile,
        nmda_gain=args.nmda_gain,
        use_nmda=args.use_nmda,
    )
    rtp = RTParams(
        beta0=args.beta0,
        beta_abs_pe=args.beta_abs_pe,
        beta_circle=args.beta_circle,
        beta_irrel_salience=args.beta_irrel_salience,
        beta_irrel_input=args.beta_irrel_input,
        rt_noise=args.rt_noise,
    )
    groups = build_groups(args)

    all_rows: List[Dict] = []
    console_metrics = [
        "kat_aberrant_salience_ms",
        "kat_unexpected_minus_expected_ms",
        "kat_circle_minus_coin_ms",
        "kat_unexpected_minus_expected",
        "kat_circle_minus_coin",
        "mean_irrel_salience",
        "irrel_minus_rel_salience",
        "mean_fired_xl_fraction",
    ]

    for gi, group in enumerate(groups):
        print(f"\n[{group.group_name}] running {args.n_agents} agents...")
        base_seed = args.seed + 10000000 * (gi + 1)
        rows, _ = simulate_one_group(
            group=group,
            n_agents=args.n_agents,
            base_seed=base_seed,
            dend=dend,
            task=task,
            fmap_params=fmap_params,
            nmda=nmda,
            rtp=rtp,
            input_mode=args.input_mode,
            save_histories=args.save_histories,
            out_dir=args.out_dir,
        )
        all_rows.extend(rows)
        for metric in console_metrics:
            vals = np.array([float(r[metric]) for r in rows], dtype=float)
            print(f"  {metric}: mean={safe_mean(vals):.4f}, sd={safe_std(vals):.4f}")

    main_metrics = [
        "kat_aberrant_salience_ms",
        "kat_shape_irrel_bias_ms",
        "kat_color_irrel_bias_ms",
        "kat_expected_logrt",
        "kat_unexpected_logrt",
        "kat_unexpected_minus_expected",
        "kat_expected_rt_ms",
        "kat_unexpected_rt_ms",
        "kat_unexpected_minus_expected_ms",
        "kat_coin_logrt",
        "kat_circle_logrt",
        "kat_circle_minus_coin",
        "kat_coin_rt_ms",
        "kat_circle_rt_ms",
        "kat_circle_minus_coin_ms",
        "mean_abs_pe",
        "mean_E",
        "mean_x_shape",
        "mean_x_color",
        "mean_irrel_input",
        "mean_rel_input",
        "mean_irrel_weight",
        "mean_rel_weight",
        "mean_abs_irrel_weight",
        "mean_abs_rel_weight",
        "mean_irrel_salience",
        "mean_rel_salience",
        "irrel_minus_rel_salience",
        "mean_fired_segment_count",
        "mean_fired_xl_fraction",
        "mean_nmda_shape",
        "mean_nmda_color",
        "xl_fraction",
        "small_fraction",
        "mean_spine_size",
        "sd_spine_size",
        "max_spine_size",
    ]

    rng_stats = np.random.default_rng(args.seed + 999)
    summary_rows = group_summary(all_rows, main_metrics, args.n_boot, rng_stats)

    anova = {}
    for metric in main_metrics:
        anova[metric] = factorial_anova_one_metric(
            rows=all_rows,
            metric=metric,
            n_perm=args.n_perm,
            rng=np.random.default_rng(args.seed + 777),
        )

    agent_csv = os.path.join(args.out_dir, "agent_metrics.csv")
    group_csv = os.path.join(args.out_dir, "group_summary.csv")
    anova_csv = os.path.join(args.out_dir, "factorial_anova.csv")
    summary_json = os.path.join(args.out_dir, "summary_all.json")

    write_csv(agent_csv, all_rows)
    write_csv(group_csv, summary_rows)
    write_csv(anova_csv, flatten_anova(anova))

    with open(summary_json, "w", encoding="utf-8") as f:
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

    print(f"\n[Saved] {agent_csv}")
    print(f"[Saved] {group_csv}")
    print(f"[Saved] {anova_csv}")
    print(f"[Saved] {summary_json}")

    if not args.no_plot:
        plot_metrics = [
            "kat_aberrant_salience_ms",
            "kat_unexpected_minus_expected",
            "kat_circle_minus_coin",
            "mean_irrel_salience",
            "irrel_minus_rel_salience",
            "mean_fired_xl_fraction",
        ]
        plot_path = os.path.join(args.out_dir, "summary_spine_weighted_isp.png")
        plot_summary_metrics(all_rows, plot_metrics, plot_path)
        print(f"[Saved] {plot_path}")

        kat3_path = os.path.join(args.out_dir, "summary_isp_rt_bar3.png")
        plot_katthagen_bar3(all_rows, kat3_path)
        print(f"[Saved] {kat3_path}")

    print("\n=== 2x2 factorial analysis: compact summary ===")
    for metric in console_metrics:
        print(f"\n[{metric}]")
        for effect, vals in anova[metric].items():
            print(
                f"  {effect}: F={vals['F']:.4f}, "
                f"p_perm={vals['p_permutation']:.4f}, "
                f"p_param={vals['p_parametric']:.4f}"
            )


if __name__ == "__main__":
    main()
