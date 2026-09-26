# Credit Default Risk Model & Approval Policy

Re-underwriting a consumer loan book of **307,511 applications** from an
emerging-market lender, where a large share of borrowers are *thin-file* — little
or no formal credit history — and roughly **8% of loans default**.

The question is not "can you predict default". It is **which applicants do you
decline, and what does declining them buy you**.

## Headline findings

**1. The application form alone is not enough.** Collapsing six one-to-many credit
tables into applicant-level features takes the model from **Gini 0.536 to 0.582**
(AUC 0.768 to 0.791), measured out-of-fold.

**2. An 80% approval cut-off lands a 4.30% bad rate**, against 8.07% if you approve
everyone — **47% fewer defaults** on a book held at four-fifths its size.

The declined fifth carries a **23.17%** bad rate, **5.4x** the approved book. The
split reconciles exactly to the portfolio rate:

```
0.80 x 4.2982%  +  0.20 x 23.1716%  =  8.0729%   <- actual default rate
```

| | Gini | AUC |
|---|---|---|
| application form only | 0.5363 | 0.7682 |
| + aggregated credit history | **0.5819** | **0.7910** |

Full numbers in [`results.json`](results.json), the policy curve in
[`threshold_sweep.csv`](threshold_sweep.csv).

## What actually drives the score

The three `EXT_SOURCE` external bureau scores dominate, as they do for everyone who
has worked this dataset. What matters here is what sits underneath them:

| rank | feature | source |
|---|---|---|
| 1-4 | `EXT_SOURCE_2/3/1`, `ORGANIZATION_TYPE` | application form |
| 5 | `CREDIT_TERM` | engineered ratio |
| 6 | `BUREAU_DEBT_RATIO_MAX` | **aggregated from bureau** |
| 8 | `INS_LATE_RATE` | **aggregated from installments** |

Two of the top eight come from the relational aggregation — the share of a prior
credit line still outstanding, and how often past installments were paid late.
Neither exists anywhere in the application form.

## Why the aggregation matters

The main application table is one row per applicant. Everything that actually
predicts default is one-to-many against it:

| table | rows | grain |
|---|---|---|
| `bureau_balance` | 27.3M | month x prior credit line |
| `installments_payments` | 13.6M | scheduled repayment |
| `POS_CASH_balance` | 10.0M | month x prior POS loan |
| `credit_card_balance` | 3.8M | month x prior card |
| `bureau` | 1.7M | prior credit line at another lender |
| `previous_application` | 1.7M | prior application to this lender |
| `application_train` | 0.3M | **the applicant** |

**~58M raw rows collapse to 307K applicant records.** That aggregation step is the
project — the modelling on top of it is comparatively routine.

## The policy, not the score

A ranked risk score is not a decision. `train_and_evaluate.py` sweeps the approval
threshold from 50% to 100% and reports the bad rate among those approved at each
point, which is the curve a credit committee actually argues over.

Everything is measured on **out-of-fold predictions** across 5 stratified folds, so
no applicant is ever scored by a model that saw their outcome.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle competition.

Downloaded via the [public mirror](https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk),
which is byte-identical and does not require accepting the competition rules.

## Method

- **Relational aggregation** — mean / max / min / sum per applicant across all six
  auxiliary tables, plus engineered ratios: debt-to-credit-line, days late per
  installment, payment-to-scheduled ratio, prior approval rate
- **LightGBM**, 5-fold stratified CV with early stopping
- **Gini** (`2 x AUC - 1`), the credit industry's standard, measured out-of-fold
- **Threshold sweep** converting score into an approval policy

AUC is computed from ranks directly rather than with scikit-learn, whose compiled
binaries are blocked by Smart App Control on the machine this was built on.

## Reproduce

```bash
pip install -r requirements.txt
kaggle datasets download -d megancrenshaw/home-credit-default-risk -p data --unzip
python build_features.py        # -> features.parquet
python train_and_evaluate.py    # -> results.json, results.png, threshold_sweep.csv
```

Seeded at 42 throughout.

## Caveats

- **This is a retrospective policy simulation on a historical loan book.** No lender
  adopted it, no applicant was declined, and no default was prevented. The output is
  a recommended cut-off, not a realised outcome.
- **Declining 20% of applicants forgoes their revenue.** The dataset carries no
  margin or pricing data, so the sweep optimises loss only. A real credit committee
  would trade bad rate against volume and lifetime value together.
- **Out-of-time validation is absent.** Folds are random, not chronological, so the
  Gini figures do not account for population drift. A production model would be
  validated on a later time window.
- Currency is unspecified in the source data, so no monetary figure is reported.
