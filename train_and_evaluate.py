"""
Train the risk model, measure what the bureau history is worth, and turn the
score into an approval policy.

Three questions, in order:

  1. How well can you rank default risk from the application form alone?
  2. How much does aggregated credit history add on top?
  3. At a given approval rate, what bad rate does the resulting policy produce?

Everything is measured on out-of-fold predictions, so no applicant is ever
scored by a model that saw their outcome.

Run:  python train_and_evaluate.py   (after build_features.py)
"""

import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_FOLDS = 5
APPROVAL_RATE = 0.80
AGG_PREFIXES = ("BUREAU_", "PREV_", "INS_", "POS_", "CC_", "BB_")

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 34,
    "max_depth": 8,
    "min_child_samples": 70,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 10.0,
    # 63 bins instead of the default 255 cuts LightGBM's internal binned
    # dataset ~4x with negligible AUC cost - this dataset was built on a
    # 16GB machine where the default caused the OS to page to disk
    "max_bin": 63,
    "force_col_wise": True,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 0,
}
N_ROUNDS = 1500
EARLY_STOP = 100


def auc(y_true, y_score):
    """Rank-based ROC AUC, equivalent to the Mann-Whitney U statistic.

    Ties get average ranks so the statistic stays exact. sklearn would do
    this in one line, but its binaries are blocked on this machine.
    """
    y_true = np.asarray(y_true)
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gini(y_true, y_score):
    return 2 * auc(y_true, y_score) - 1


def stratified_folds(y, n_folds, seed):
    """Fold indices preserving the class ratio, without sklearn."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def train_oof(X, y, label):
    """Out-of-fold predictions across N_FOLDS.

    The full dataset is binned once and each fold takes a `subset` of it.
    Building a fresh Dataset per fold would copy 80% of the feature matrix
    five times over, which this machine does not have the RAM for.
    """
    folds = stratified_folds(y.to_numpy(), N_FOLDS, SEED)
    oof = np.zeros(len(y))
    iters = []

    full = lgb.Dataset(X, y, free_raw_data=False)
    full.construct()

    for k in range(N_FOLDS):
        train_idx = np.flatnonzero(folds != k)
        valid_idx = np.flatnonzero(folds == k)

        dtrain = full.subset(train_idx)
        dvalid = full.subset(valid_idx)

        booster = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=N_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        oof[valid_idx] = booster.predict(
            X.iloc[valid_idx], num_iteration=booster.best_iteration
        )
        iters.append(booster.best_iteration)
        print(f"    fold {k + 1}/{N_FOLDS}  gini {gini(y.iloc[valid_idx], oof[valid_idx]):.4f}"
              f"  ({booster.best_iteration} rounds)", flush=True)

        del dtrain, dvalid
        gc.collect()

    print(f"  {label:<22} GINI {gini(y, oof):.4f}   AUC {auc(y, oof):.4f}", flush=True)
    del full
    gc.collect()
    return oof, int(np.mean(iters)), booster


def threshold_sweep(y, score):
    """For each approval rate, the bad rate among those approved."""
    order = np.argsort(score)          # lowest predicted risk first
    y_sorted = np.asarray(y)[order]
    cum_bads = np.cumsum(y_sorted)
    n = len(y_sorted)

    rates = np.arange(0.50, 1.001, 0.01)
    rows = []
    for r in rates:
        k = max(int(round(r * n)), 1)
        rows.append({
            "approval_rate": r,
            "approved": k,
            "bad_rate": cum_bads[k - 1] / k,
            "declined_bad_rate": (cum_bads[-1] - cum_bads[k - 1]) / max(n - k, 1),
        })
    return pd.DataFrame(rows)


def plot(sweep, base_rate, chosen, gini_base, gini_full):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(sweep.approval_rate * 100, sweep.bad_rate * 100, color="#4c72b0", lw=2)
    ax1.axhline(base_rate * 100, color="#8c8c8c", ls="--", lw=1.2,
                label=f"approve everyone: {base_rate:.2%}")
    ax1.scatter([APPROVAL_RATE * 100], [chosen * 100], color="#c44e52", zorder=5, s=60,
                label=f"{APPROVAL_RATE:.0%} approval: {chosen:.2%}")
    ax1.set_xlabel("approval rate (%)")
    ax1.set_ylabel("bad rate among approved (%)")
    ax1.set_title("Approval policy trade-off")
    ax1.legend(frameon=False, fontsize=9)

    ax2.bar(["application\nform only", "+ aggregated\ncredit history"],
            [gini_base, gini_full], color=["#8c8c8c", "#4c72b0"], width=0.55)
    for i, v in enumerate([gini_base, gini_full]):
        ax2.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=11)
    ax2.set_ylabel("Gini (out-of-fold)")
    ax2.set_ylim(0, gini_full * 1.25)
    ax2.set_title("What the relational aggregation is worth")

    fig.tight_layout()
    fig.savefig("results.png", dpi=150)
    print("\nwrote results.png")


def main():
    print("Loading features\n" + "=" * 46)
    try:
        df = pd.read_parquet("features.parquet")
    except (ImportError, FileNotFoundError):
        df = pd.read_pickle("features.pkl")
    y = df.pop("TARGET")

    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype("category")
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

    base_cols = [c for c in df.columns if not c.startswith(AGG_PREFIXES)]
    print(f"applicants               {len(df):,}")
    print(f"default rate             {y.mean():.4%}")
    print(f"features (application)   {len(base_cols):,}")
    print(f"features (all)           {df.shape[1]:,}")

    print("\nBaseline: application form only\n" + "=" * 46, flush=True)
    base = df[base_cols]
    oof_base, _, _ = train_oof(base, y, "application only")
    del base
    gc.collect()

    print("\nFull: + aggregated credit history\n" + "=" * 46, flush=True)
    oof_full, _, booster = train_oof(df, y, "with aggregations")

    gini_base, gini_full = gini(y, oof_base), gini(y, oof_full)

    print("\nApproval policy\n" + "=" * 46)
    sweep = threshold_sweep(y, oof_full)
    row = sweep.loc[(sweep.approval_rate - APPROVAL_RATE).abs().idxmin()]
    base_rate = y.mean()
    reduction = (base_rate - row.bad_rate) / base_rate

    print(f"approve everyone         {base_rate:.4%} bad rate")
    print(f"approve lowest-risk {APPROVAL_RATE:.0%}  {row.bad_rate:.4%} bad rate")
    print(f"declined {1-APPROVAL_RATE:.0%}             {row.declined_bad_rate:.4%} bad rate")
    print(f"reduction in defaults    {reduction:.2%}")
    print(f"separation               {row.declined_bad_rate / row.bad_rate:.2f}x")

    print("\nHeadline\n" + "=" * 46)
    print(f"Gini  {gini_base:.4f} -> {gini_full:.4f}   (+{gini_full - gini_base:.4f})")
    print(f"AUC   {auc(y, oof_base):.4f} -> {auc(y, oof_full):.4f}")
    print(f"{APPROVAL_RATE:.0%} approval cut-off lands a {row.bad_rate:.2%} bad rate")

    sweep.to_csv("threshold_sweep.csv", index=False)
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp.to_csv("feature_importance.csv", index=False)

    json.dump({
        "applicants": int(len(df)),
        "portfolio_default_rate": float(base_rate),
        "gini_application_only": float(gini_base),
        "gini_with_aggregations": float(gini_full),
        "approval_rate": float(APPROVAL_RATE),
        "bad_rate_at_cutoff": float(row.bad_rate),
        "declined_bad_rate": float(row.declined_bad_rate),
        "default_reduction": float(reduction),
    }, open("results.json", "w"), indent=2)

    print("wrote threshold_sweep.csv, feature_importance.csv, results.json")
    plot(sweep, base_rate, row.bad_rate, gini_base, gini_full)


if __name__ == "__main__":
    main()
