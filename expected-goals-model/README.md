# Expected Goals (xG) model from scratch

A single Jupyter notebook that builds an Expected Goals model end to end, from
pulling raw StatsBomb event data to a fitted, calibrated model and a set of
diagnostic plots. The notebook runs top to bottom on a fresh kernel with no
manual steps.

The approach follows David Sumpter's teaching order: start with the simplest
model that a person can actually interpret, a logistic regression on shot
distance and angle, understand it fully, and only then add complexity. Calibration
is treated as a real result rather than an afterthought. A model whose predicted
xG does not match observed scoring rates is not useful, however high its AUC.

## What the notebook does

1. Pulls StatsBomb open data through the `mplsoccer` Sbopen parser (no login, no
   paid API) for four recent men's international tournaments: World Cup 2018,
   World Cup 2022, Euro 2020, and Euro 2024.
2. Filters to shots, drops penalties, and builds the goal/no-goal target after
   confirming the exact outcome label in the data.
3. Engineers features from raw shot location: distance to goal centre and the
   angle subtended by the two goalposts, both derived from first principles, plus
   body part, play pattern, shot situation, and context flags (under pressure,
   first-time, one-on-one). Freeze-frame data adds the number of defenders in the
   shot-to-goal cone and the goalkeeper's distance from goal.
4. Fits three models: a baseline logistic regression on distance and angle, an
   extended logistic regression with the context features, and an XGBoost model
   on the full feature set with lightly tuned, group-aware cross-validation.
5. Evaluates on a held-out set split by match (not by shot, to avoid leakage),
   reporting log loss, Brier score, and ROC-AUC, plus calibration curves and an
   aggregate check of summed xG against actual goals.
6. Benchmarks every model against StatsBomb's own per-shot xG.
7. Draws the shot map, the xG surface across the pitch, and the coefficient and
   feature-importance plots with `mplsoccer` and `matplotlib`.

## Headline results

Test set (matches unseen in training), 1,334 shots:

| Model | Log loss | Brier | ROC-AUC |
|---|---|---|---|
| Baseline logistic (distance + angle) | 0.2645 | 0.0736 | 0.746 |
| Extended logistic (+ context) | 0.2475 | 0.0698 | 0.796 |
| XGBoost (full features) | 0.2414 | 0.0680 | 0.814 |
| StatsBomb xG (reference) | 0.2313 | 0.0652 | 0.828 |

The two-feature baseline already captures most of the signal and is well
calibrated. The context features give a real but secondary gain, and the
freeze-frame features are what let the boosted model close most of the remaining
distance to StatsBomb, because they encode who is standing between the ball and
the net. Full numbers and interpretation are in [RESEARCH_NOTE.md](RESEARCH_NOTE.md).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name xg-venv --display-name "Python (xg-venv)"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=xg-venv \
  xg_model_from_scratch.ipynb
```

Or open `xg_model_from_scratch.ipynb` in Jupyter or VS Code and run all cells.

Notes on reproducibility:

- Random seeds are set at the top of the notebook.
- The first run downloads roughly 230 matches of event data and takes about a
  minute. Shots are cached to a local `data_cache/` folder so later runs are
  fast. That folder and the saved model in `models/` are generated on run and are
  not committed; delete `data_cache/` to force a fresh pull.
- The numpy, scipy, and pandas versions in `requirements.txt` must stay together.
  An older numpy against these scipy and pandas builds fails to import with an
  ABI error.

## Files

- `xg_model_from_scratch.ipynb`: the complete notebook, saved with its outputs
  and plots so the results are visible on GitHub without running anything.
- `RESEARCH_NOTE.md`: methodology and results write-up.
- `requirements.txt`: pinned dependency versions.

## Data credit

All data is StatsBomb free open data (https://github.com/statsbomb/open-data),
accessed through `mplsoccer`. StatsBomb requires that their data be attributed.
Pitch dimensions and goalpost coordinates follow the StatsBomb Open Data
Specification in that repository.
