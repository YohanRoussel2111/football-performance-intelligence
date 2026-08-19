# Football Performance Intelligence

**From data to actionable performance decisions.**

A demonstrable, end-to-end platform that turns complex longitudinal football
performance data into **explainable, actionable** intelligence for the people who
make availability and load decisions: physical performance, sports science,
sports medicine, football intelligence and the coaching staff.

The project is built around one question:

> **How can we turn complex longitudinal football performance data into
> explainable, actionable intelligence that helps practitioners make better
> decisions about player availability and performance?**

It deliberately covers the *whole* pipeline rather than a single pretty
dashboard:

```
Data → Data Engineering → Feature Engineering → Statistical Analysis →
Machine Learning → Explainable AI → Decision Support → Monitoring
```

> ⚠️ **Read this first.** The platform runs on **synthetic** data and is a
> **decision-support** demonstration. Model outputs are an operational
> early-warning signal intended to prompt a **staff review** — they are **not**
> a medical diagnosis and **not** an injury prediction. See *Limitations* and
> *Ethical considerations*.

---

## 1. Project overview

A performance department collects data from many streams — external load (GPS),
internal load (sRPE, HR), neuromuscular tests (CMJ, force), wellness
questionnaires, match exposure and availability. Individually these streams are
noisy; together, viewed *longitudinally and per-player*, they carry signal about
when an athlete is drifting toward a state that warrants closer monitoring.

This application ingests those streams, engineers longitudinal features against
each player's **own baseline**, scores a calibrated machine-learning model, and —
crucially — **explains every flag** in practitioner language, ending in a cautious
review recommendation rather than an automated decision.

The app has nine working areas: Executive Brief, Squad Overview, Player Profile,
Load Monitoring, Availability, AI Risk / Monitoring, Performance Actions,
Scenario Simulator, and Model Performance / Model Monitoring / Data Quality.

## 2. Problem statement

Availability is one of the strongest determinants of results in professional
football, and the cost of a misjudged training load is high. Staff are not short
of data; they are short of *time* to fuse many longitudinal streams into a clear,
trustworthy, individualised picture before the next session. The operational task
this project models is an **early-warning triage**: given everything known about a
player up to today, how much attention does this player warrant over the coming
days — and, above all, **why**? Framing it as triage (not diagnosis) keeps the
tool inside the boundary where a data-driven system genuinely helps a
multidisciplinary staff.

## 3. Architecture

The code is layered so each stage is independent and testable, and so the two
"industrialisation seams" (database and front-end) can be swapped without
rewriting the analytics.

```
football-performance-intelligence/
├── app/                      # Streamlit front-end
│   ├── main.py               # entry point (navigation + demo controls)
│   ├── app_core.py           # cached data/model access (single source of truth)
│   ├── ui.py                 # shared styling & components
│   └── views/                # one module per page
├── src/
│   ├── config.py             # paths, seed, thresholds, target & leakage rules
│   ├── data_processing/      # generation, pipeline (validate/clean/aggregate), SQLite
│   ├── feature_engineering/  # longitudinal features + individual baselines
│   ├── analytics/            # dimensional monitoring + Performance Monitoring Index
│   ├── ml/                   # dataset/target, training, calibration, explain, inference
│   └── visualization/        # Plotly theme & chart library
├── sql/                      # example analytical queries
├── tests/                    # pytest suite (pipeline, leakage, temporal, model)
├── data/                     # SQLite db + exported raw CSVs (generated)
├── models/                   # trained model bundle + metrics.json (generated)
├── build_demo.py             # one-shot offline build
├── run_app.py                # launcher
└── requirements.txt
```

**Portability by design.** The persistence layer (`src/data_processing/database.py`)
is URI-driven (`sqlite:///…`), so a PostgreSQL engine can be introduced at a single
seam. The analytics/ML layers never touch the front-end, so Streamlit can be
replaced by a service API without changing a line of modelling code.

## 4. Data model

The SQLite schema is normalised into the tables a club would actually keep:

| Table | Grain | Contents |
|---|---|---|
| `players` | player | id, name, age, position, dominant leg, anthropometrics |
| `matches` | fixture | date, competition, opponent, home/away, congestion flag |
| `sessions` | player-day | session type, matchday code (MD-x), sRPE, minutes |
| `gps_data` | player-day | total distance, HSR, sprint distance, accel/decel, player/metabolic load |
| `wellness` | player-day | sleep duration/quality, fatigue, soreness, stress, mood |
| `physical_tests` | player-test | CMJ height & contraction time, eccentric/concentric/peak force, 10 m/30 m sprint |
| `availability` | player-day | status (available / modified / unavailable), minutes, days since match |
| `availability_episodes` | episode | start, end, type, duration |
| `predictions` | player-day | model version, risk probability, monitoring level |

**Wellness convention** (documented and used consistently): higher
fatigue/soreness/stress = worse; higher sleep quality/mood/duration = better.

The data itself is **synthetic**, produced by a seeded *latent-hazard* simulator
(`generate_synthetic.py`): a hidden physiological state (accumulated fatigue,
acute:chronic load, neuromuscular fatigue, sleep debt, individual susceptibility)
drives both the observable measurements (with realistic noise and missingness)
and a daily hazard of entering a reduced-availability episode. The model only
ever sees the noisy observables, so the signal is genuinely learnable but far
from perfect — which is the point.

## 5. Feature engineering

Every feature is computed **causally** (only data up to and including day *t*) and,
wherever it matters, **relative to the player's own history** rather than the squad
mean. The families:

- **Load** — rolling sums/means over 3/7/28 days, 7-day mean & SD, **monotony**
  (mean ÷ SD), **strain** (weekly load × monotony), and the **acute:chronic
  workload ratio** (7:28). ACWR is reported with explicit caution: the coupled
  rolling-average form is noisy, so the raw acute and chronic windows are exposed
  alongside it and ACWR is never used as a hard rule.
- **Wellness** — 7-day rolling averages, deviation from an **individual expanding
  baseline**, and individual **z-scores**.
- **Neuromuscular** — CMJ carried forward between test days, percentage change and
  z-score versus the player's baseline, and a 7-day trend.
- **Match congestion** — minutes in the last 7 and 14 days, matches in the last 14
  and 28 days, and days since the last match.
- **Static context** — age (a legitimate, observable risk modifier).

The individual-baseline philosophy is central: a 3 cm CMJ drop means something
different for a player who habitually jumps 42 cm than for one at 30 cm, and only
the *within-athlete* comparison catches it.

## 6. Machine learning methodology

**Target (leakage-safe).** On a day when a player is *currently available*, predict
whether a reduced-availability event (modified training **or** unavailable) begins
within the next **7 days**. Restricting to currently-available days means the model
learns to anticipate a *transition*, not to detect an ongoing episode.

**Candidates.** Three models spanning the bias/variance spectrum are trained and
compared: **Logistic Regression** (scaled, class-weighted — the interpretable
baseline), **Random Forest** (class-weighted), and **XGBoost**
(`scale_pos_weight`). Class imbalance is handled at the estimator level rather than
by resampling, so calibration stays meaningful.

**Selection & results.** Models are ranked by **temporal cross-validation PR-AUC**
(the right metric for rare events, estimated on expanding-window folds). In the
reference build Logistic Regression was selected. Indicative held-out **test**
performance (calibrated):

| Metric | Value |
|---|---|
| ROC-AUC (validation / test) | 0.71 / 0.66 |
| PR-AUC (test) | 0.33 (base rate 0.23) |
| Recall / Precision (test) | 0.69 / 0.31 |
| Brier (calibrated / raw) | **0.176 / 0.247** |

The operating threshold is tuned for **recall**: a screening tool should rarely
miss a genuine case, accepting some false alarms that staff can quickly rule out.
The visible validation→test drop is the honest cost of **non-stationarity** (a
late-season congested block raises the base rate) — exactly what deployment faces.

## 7. Temporal validation

The season is split **strictly by calendar time** — no shuffling — into the first
70 % (train), the next 15 % (validation/calibration) and the final 15 % (test).
A random train/test split is inappropriate here for two reasons: it leaks *future*
information into training, and it scatters repeated measurements of the *same
player* across both sides, so the model is effectively tested on players it has
already partly memorised. Time-based splitting mirrors reality — you always
predict the future from the past — and yields an honest, if lower, estimate of
deployed performance. Cross-validation during model comparison uses the same
principle (expanding-window folds), never a shuffled K-fold.

## 8. Explainability

Explainability is a **core feature, not an add-on**. Every flag can be expanded
into "here is *why*":

- **SHAP** values give each feature's signed contribution for a specific
  player-day, using the fast exact explainer for the model family (Linear for the
  logistic model, Tree for tree models).
- A **translation layer** maps raw features into practitioner language ("Reduced
  CMJ vs individual baseline", "Short recovery window", "High-speed running load
  up"), keeping only the drivers that actually move *this* player's risk.

This is what turns *"the model says HIGH"* into *"the model says HIGH because
recent high-speed load is up, CMJ is below this player's baseline and reported
fatigue is elevated — please review before the next high-intensity session."*
The interface makes the chain legible: **DATA → SIGNAL → MODEL → EXPLANATION →
ACTION**.

## 9. Calibration

A risk number is only useful for decisions if it means what it says. The selected
model is **calibrated** (Platt scaling) on the validation window and evaluated on
the untouched test window. Calibration lowered the test **Brier score from 0.247
to 0.176**, so a "30 % risk" corresponds to events happening roughly 30 % of the
time. The reliability curve and Brier score are shown on the *Model Performance*
page. Calibration matters operationally because staff act on the *level* of a
probability, not just its rank order.

## 10. Model monitoring

The *Model Monitoring* page treats the model as a production system: it shows the
model version and training date, the number of training observations, headline
metrics, **feature drift** (standardised shift of recent feature means versus the
training distribution), the **prediction distribution**, and data
freshness/completeness. A simple **degradation flag** fires when drift, staleness,
completeness or a collapsed prediction distribution breach tolerance — the trigger
to retrain or investigate.

## Avoiding data leakage

Leakage is the fastest way to build a model that looks brilliant offline and fails
in the building. The safeguards here are explicit:

- **No post-outcome features.** Any variable only knowable at or after the target
  event (the availability labels themselves, anything derived from them) is on a
  **blocklist** in `config.py` and asserted out of the feature set in
  `ml/dataset.py` (`assert_no_leakage`). A unit test proves the guard raises if a
  blocklisted column is ever passed.
- **Causal features only.** All rolling/baseline features use data up to day *t*;
  a test checks that the 7-day rolling load equals the sum of the trailing ≤7 days
  (no peeking ahead).
- **Future-only target.** The label at day *t* depends solely on availability in
  *t+1…t+H*; a test verifies this against the raw event series.
- **Temporal split & CV.** No shuffled splitting, so neither future time nor
  repeated within-player measurements leak across the train/test boundary.
- **Calibration on a held-out window**, then evaluation on a *third* untouched
  window.

## 11. Limitations

This project is a rigorous **demonstration**, and it is important to be precise
about what it is not. The data is **synthetic**: there are **no real medical
records** and no clinically validated injury outcomes, so absolute performance
numbers describe the simulator, not the real world. The model is **demonstrative**
— its predictions are an aid to decision-making, never a certainty, and
**correlation is not causation**: a feature that raises the model's score is not
thereby a cause of anything. Synthetic relationships, however carefully built,
cannot capture the full complexity of a squad, and any real deployment would carry
**risk of bias** across positions, ages and individuals. Before any real use the
system would require **external validation on real, multi-club data** and
**clinical/performance sign-off**. The tool must never be presented as able to
predict injuries with certainty.

## 12. Ethical considerations

Availability data is sensitive and adjacent to medical information. The design
choices reflect that: the system recommends a **staff review**, never a training,
selection or medical decision; it never outputs instructions such as "do not
train" or "this player will get injured"; explanations are provided so decisions
stay with accountable humans who can weigh context the model cannot see; and
individual baselines are used to avoid penalising players for stable individual
differences. In production, access control, data minimisation, athlete
transparency and a clear human-in-the-loop policy would be prerequisites, not
afterthoughts.

## 13. How to run

Requires Python 3.10+.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. build the demo artefacts (synthetic data → SQLite, train & calibrate model)
python build_demo.py

# 3. launch the app
python run_app.py            # or:  streamlit run app/main.py
```

`run_app.py` builds the artefacts automatically on first launch if they are
missing. In the app, the sidebar **Demo controls** reproduce the flow live:
**1 · Load Demo Dataset → 2 · Run Model → 3 · Generate Predictions**, then open
**AI Risk / Monitoring → Explain a player**.

Run the tests with:

```bash
pytest -q
```

**Environment tested:** Python 3.11 · pandas 3.0 · numpy 2.4 · scikit-learn 1.8 ·
xgboost 3.2 · shap 0.51 · streamlit 1.61 · plotly 6.9.

## 14. Future development

Natural next steps toward a club-grade system: replace SQLite with PostgreSQL and
the synthetic feed with real vendor integrations (Catapult/STATSports GPS,
wellness apps, force-plate systems) behind the existing URI/ingestion seams; move
scoring behind a service API with a scheduled feature store and model registry;
add survival / time-to-event models and hierarchical (mixed-effects) models that
formalise the individual-baseline idea; introduce drift-triggered retraining and
champion/challenger evaluation; and — most importantly — run a **prospective,
externally validated** study with medical and performance staff before any
operational reliance. The architecture here is intended to make each of those a
localised change rather than a rewrite.

---

*Built as a portfolio/interview demonstration of the full performance-data
pipeline: scientific validity, technical quality, explainability and practitioner
usability over "AI gadget" appeal.*
