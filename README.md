# Schizophrenia computational model

Multi-level computational model linking dendritic spine pathology and
dopaminergic negative-prediction-error dysfunction to schizophrenia-like
aberrant salience and reversal-learning deficits.

## Contents

- `run_isp_stas_0625_bar3_largefont.py` — Implicit Salience Paradigm (ISP)
  task simulation. Spine-weighted associative-learning model; computes the
  aberrant-salience RT score for HC/Spine/DA/SZ groups, including a 2×2
  factorial ANOVA (parametric + permutation) on the simulated agents.
- `run_waltz_0625_bar4_largefont.py` — Probabilistic reversal-learning task
  simulation (Waltz & Gold–style paradigm). Same spine-weighted mechanism;
  computes initial discriminations, reversals, and error rates for the same
  four groups, with the same factorial ANOVA.
- `run_neuro2026_results_and_sensitivity1_2.py` — Runs both models together
  across a grid of the composite `spine_pathology_strength` and
  `negative_learning_scale` parameters and produces the baseline bar charts
  and sensitivity curves.

## Requirements

```
pip install numpy scipy matplotlib
```

## Example usage

```bash
python3 run_isp_stas_0625_bar3_largefont.py --out_dir out_isp --n_agents 50
python3 run_waltz_0625_bar4_largefont.py --out_dir out_waltz --n_agents 50

python3 run_neuro2026_results_and_sensitivity1_2.py \
  --isp_model run_isp_stas_0625_bar3_largefont.py \
  --waltz_model run_waltz_0625_bar4_largefont.py \
  --out_dir neuro2026_results_sensitivity --n_agents 50
```

Each script writes agent-level and group-summary CSVs plus figures to
`--out_dir`. Run with `--help` for the full list of biophysical, learning,
and task parameters.
