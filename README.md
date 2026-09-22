# Schizophrenia computational model

Multi-level computational model linking dendritic spine pathology and
dopaminergic negative-prediction-error dysfunction to schizophrenia-like
aberrant salience and reversal-learning deficits.

## Contents

- `EMBC_isp.py` — Implicit Salience Paradigm (ISP) task simulation.
  Spine-weighted associative-learning model; computes the aberrant-salience
  RT score for HC/Spine/DA/SZ groups, including a 2×2 factorial ANOVA
  (parametric + permutation) on the simulated agents.
- `EMBC_waltz.py` — Probabilistic reversal-learning task simulation
  (Waltz & Gold–style paradigm). Same spine-weighted mechanism; computes
  initial discriminations, reversals, and error rates for the same four
  groups, with the same factorial ANOVA.
- `EMBC_analysis.py` — Runs both models together across a grid of the
  composite `spine_pathology_strength` and `negative_learning_scale`
  parameters and produces the baseline bar charts and sensitivity curves.

## Requirements

```
pip install numpy scipy matplotlib
```

## Example usage

```bash
python3 EMBC_isp.py --out_dir out_isp --n_agents 50
python3 EMBC_waltz.py --out_dir out_waltz --n_agents 50

python3 EMBC_analysis.py \
  --isp_model EMBC_isp.py \
  --waltz_model EMBC_waltz.py \
  --out_dir neuro2026_results_sensitivity --n_agents 50
```

Each script writes agent-level and group-summary CSVs plus figures to
`--out_dir`. Run with `--help` for the full list of biophysical, learning,
and task parameters.
