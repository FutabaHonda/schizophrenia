# Final calibrated parameters (EMBC2027 draft, checkpoint 2026-09-24)

This documents the exact parameter set used to produce the EMBC2027 draft
paper's reported results, as a durable reference independent of the chat
session. See `README.md` for general usage.

## ISP task (`EMBC_isp.py`)

```
python3 EMBC_isp.py \
  --out_dir <out> --n_agents 50 --seed 0 \
  --alpha_neg_scale_da 0.50 \
  --beta_irrel_salience 0.0 --beta_irrel_input 0.0 \
  --beta0 6.22 --beta_circle 0.075 --beta_abs_pe 0.094 \
  --n_perm 2000 --n_boot 50
```

- `beta0=6.22`: Katthagen et al. (2018) HC-fitted RT-equation intercept.
  Provably invariant to Cohen's d for any within-agent RT contrast (log-RT is
  additive in beta0, RT is its exponential, so beta0 is a common
  multiplicative factor that cancels in both the mean difference and pooled
  SD of the standardized effect size). Verified: d=0.4023 identical at
  beta0=6.18 and beta0=6.22, all else fixed.
- `beta_circle=0.075`: average of Katthagen et al.'s reported HC/SZ
  circle-trial coefficients (0.08/0.07). NOT d-invariant (d=0.4023 at
  beta_circle=0.05 vs. d=0.2924 at beta_circle=0.08, all else fixed), so it
  was fixed to this literature value rather than left at the original
  heuristic default (0.05).
- `beta_abs_pe=0.094`: searched with beta0/beta_circle/beta_irrel_* fixed as
  above, against target d(SZ,HC)=0.39 (Katthagen et al.'s raw group means).
  Search points: 0.08->d=0.333, 0.10->d=0.415, 0.12->d=0.490; linear
  interpolation selected 0.094; confirmed by direct simulation, d=0.3916.
- `alpha_neg_scale_da=0.50` (gamma_neg): largest value that keeps all three
  Waltz & Gold (2007) categorical chi-square tests significant while
  improving on the original default's (0.35) oversized magnitude.

## Waltz (reversal-learning) task (`EMBC_waltz.py`)

```
python3 EMBC_waltz.py --out_dir <out> --n_agents 50 --seed 0 \
  --alpha_neg_scale_da 0.50
```

All other flags at script defaults.

## Results at this parameter set (n=50 agents/group, seed=0)

| Group | Aberrant salience (mean+-SD) | Total reversals (mean) |
|---|---|---|
| HC | 4.02 +- 1.86 | 2.52 |
| Spine | 4.57 +- 2.79 | 3.16 |
| DA | 5.57 +- 3.02 | 1.30 |
| SZ | 4.92 +- 2.69 | 0.86 |

- d(SZ,HC) aberrant salience = 0.39 (lit. d=0.39, Katthagen et al.)
- Total reversals: chi2(6)=32.5, p<.0001, Cramer's V=0.57 (lit. V=0.51, Waltz & Gold)
- Full statistics (ANOVA table, chi-square/Welch-t tests, all six RQ1
  measures) are in the paper draft, Tables III-V.

## Known, disclosed open issue at this checkpoint

A diagnostic check found the spine-generation mechanism (lognormal size
distribution + XL clustering + segment-firing threshold) does NOT
quantitatively reproduce two specific findings of Obi-Nagata et al. (2023,
Sci. Adv., eade5973), despite correctly using their XL-spine diameter cutoff
(0.8 um):
1. XL-spine fraction is ~26-28% of all spines in this model (both HC and
   pathological conditions), vs. a small single-digit percentage reported in
   the source paper (and barely differs between conditions, vs. their
   reported several-fold disease increase).
2. The model shows no meaningful difference in the number of co-active
   spines needed to cross the firing threshold between XL-dominated and
   typical-dominated dendritic segments (~2.0-2.6 either way), vs. the
   source paper's reported ~3 XL spines vs. ~8 typical spines (~2.7x
   threshold reduction).

An attempt to recalibrate the spine-generation/firing mechanism to fix this
is in progress as of this checkpoint. This file, plus the full working state
saved in the chat session's scratchpad
(`checkpoint_pre_spine_recalibration/`), is the fallback if that attempt
does not complete before the submission deadline.
