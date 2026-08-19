# Interview demo — 5-minute script

A tight, five-minute walkthrough that shows you understand the **whole** pipeline,
not just a dashboard. Practise the transitions; let the *explainability* moment
land — that is the differentiator.

**Before you start:** `python run_app.py`, open the app, and have the sidebar
**Demo controls** visible. The default landing page is the **Executive Brief**.

One sentence to open with:

> *"This turns longitudinal performance data into explainable, actionable
> intelligence about player availability — and, importantly, it always shows the
> staff **why**, so the decision stays with them."*

---

## 30 seconds — the problem

Stay on the **Executive Brief**.

- "Availability drives results, and staff aren't short of data — they're short of
  time to fuse many longitudinal streams before the next session."
- "This page answers, in under 30 seconds: **who needs attention, why, what
  changed, what to review, and how confident we are.** Everything else in the app
  is the evidence behind it."
- Point to the counts (available / to-monitor / modified / unavailable) and the
  priority review list.

## 1 minute — the data

Go to **Squad Overview**, then **Data Quality**.

- "It's one professional season for 28 players: GPS external load, internal load,
  wellness, CMJ and force tests, match exposure and availability — modelled with a
  realistic latent-hazard simulator, because real club and medical data is
  proprietary and sensitive."
- Open **Data Quality**: "The pipeline validates, de-duplicates, flags and caps
  physiological outliers *non-destructively*, and reports completeness and
  freshness. If the inputs aren't trustworthy, the page says so — you don't run a
  model on bad data."
- One line on feature engineering: "Features are built **per player against their
  own baseline** — a CMJ drop means something different for every athlete — and
  strictly causally, using only the past."

## 1 minute — the model

Go to **Model Performance**.

- "The task is leakage-safe: on a day a player is *available*, will he enter a
  reduced-availability state within 7 days? I compare logistic regression, random
  forest and XGBoost."
- "The split is **temporal**, not random — you always predict the future from the
  past, and a random split would leak future data and repeated player measurements.
  I select on **temporal-CV PR-AUC** because events are rare."
- Point to the numbers: "Validation ROC-AUC ~0.71, honest test ~0.66 — the drop is
  real non-stationarity from a congested block, which is what deployment faces.
  The threshold is tuned for **recall**: a screening tool shouldn't miss cases."
- Point to **calibration**: "And it's **calibrated** — Brier drops from 0.25 to
  0.18 — so a 30 % risk really means ~30 %. That matters when staff act on the
  number."

## 1 minute — the explainability (the moment)

Go to **AI Risk / Monitoring → Explain a player** (it defaults to the
highest-risk player).

- "This is the core of the project. The model doesn't just say HIGH."
- Walk the **SHAP** bar chart: "For *this* player, on *this* day, risk is driven by
  recent high-speed load, CMJ below his own baseline, and elevated fatigue vs his
  baseline — each factor's contribution, in plain language."
- Read the **DATA → SIGNAL → MODEL → EXPLANATION → ACTION** panel aloud.
- "So we move from *'the model says high risk'* to *'here's exactly why'* — that's
  what earns a practitioner's trust."

## 1 minute — the operational decision

Go to **Performance Actions**, then **Scenario Simulator**.

- **Performance Actions**: "The explanation becomes cautious, **non-medical** review
  prompts — 'review recent external load', 'compare CMJ with individual baseline',
  'discuss with performance staff'. It recommends a **review**; it never says 'don't
  train' or 'he'll get injured'. The decision stays human."
- **Scenario Simulator**: "Staff can explore — drop next-match minutes, add sleep,
  lower load — and see how the model's read changes. I present this as an
  **exploratory sensitivity** on the model, explicitly *not* causal inference."

## 30 seconds — from prototype to club infrastructure

Close on the architecture (or the **Model Monitoring** page).

- "Everything's built on two swap-in seams: the store is URI-driven, so SQLite
  becomes PostgreSQL at one point; and the analytics never touch the front-end, so
  Streamlit becomes a service API without changing the modelling."
- "In a club I'd wire real GPS/wellness/force-plate feeds into the same ingestion,
  add a feature store and model registry, **drift-triggered retraining** — the
  Model Monitoring page is the start of that — and, before any operational
  reliance, run a **prospective, externally validated** study with medical and
  performance staff."
- Final line: *"The priority throughout was scientific validity, explainability and
  practitioner usability — not an AI gadget."*

---

### If you have 30 extra seconds / likely questions

- **"Why is performance not higher?"** — "Because I refused to leak. The honest
  number is the deployable number; I can show you the leakage guards and the
  temporal split in the code and tests."
- **"Is this predicting injuries?"** — "No. It's an availability early-warning to
  prompt review. It's synthetic data and a decision-support tool, not a medical
  device — that boundary is deliberate and documented."
- **"How do you handle a new signing with no history?"** — "Baselines are
  expanding with a warm-up period; until stable, the squad-level features and
  dimensional rules carry more weight, and the app flags low individual history."
- **"What would break in production first?"** — "Data quality and drift — which is
  exactly why both have their own monitoring pages."
