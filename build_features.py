"""
Collapse the Home Credit relational tables into one row per applicant.

The main application table is one row per loan application. Everything that
actually predicts default - prior credit-bureau records, previous applications,
repayment history - is one-to-many against it. This script aggregates each of
those tables down to applicant level and joins them on SK_ID_CURR.

Tables are loaded and released one at a time; loading all six at once needs
roughly 10GB of RAM.

Run:  python build_features.py
Out:  features.parquet  (one row per applicant)
"""

import gc
import numpy as np
import pandas as pd

DATA = "data"
AGGS = ["mean", "max", "min", "sum"]
row_counts = {}


def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    row_counts[name] = len(df)
    print(f"  {name:<24} {len(df):>12,} rows  x {df.shape[1]:>3} cols")
    return df


def numeric_agg(df, key, prefix, extra=None):
    """Aggregate every numeric column by `key`, flatten the column names."""
    drop = [c for c in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU") if c != key]
    numeric = df.select_dtypes("number").drop(columns=drop, errors="ignore")
    if key not in numeric.columns:
        numeric[key] = df[key]

    out = numeric.groupby(key).agg(AGGS)
    out.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in out.columns]
    out[f"{prefix}_COUNT".upper()] = df.groupby(key).size()

    if extra is not None:
        out = out.join(extra)
    return out


def bureau_features():
    """Bureau records, plus their monthly balance history rolled up first."""
    print("\nbureau + bureau_balance")
    bb = load("bureau_balance")
    bb["IS_DPD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_DPD_MONTHS=("IS_DPD", "sum"),
    )
    del bb
    gc.collect()

    bureau = load("bureau")
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU")
    del bb_agg
    gc.collect()

    # share of the reported credit line still outstanding
    bureau["DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)

    active = (
        bureau.assign(IS_ACTIVE=(bureau["CREDIT_ACTIVE"] == "Active").astype("int8"))
        .groupby("SK_ID_CURR")["IS_ACTIVE"]
        .agg(BUREAU_ACTIVE_N="sum", BUREAU_ACTIVE_RATE="mean")
    )

    out = numeric_agg(bureau, "SK_ID_CURR", "BUREAU", extra=active)
    del bureau, active
    gc.collect()
    return out


def previous_application_features():
    print("\nprevious_application")
    prev = load("previous_application")
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)

    approved = (
        prev.assign(IS_APPROVED=(prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int8"))
        .groupby("SK_ID_CURR")["IS_APPROVED"]
        .agg(PREV_APPROVED_N="sum", PREV_APPROVAL_RATE="mean")
    )

    out = numeric_agg(prev, "SK_ID_CURR", "PREV", extra=approved)
    del prev, approved
    gc.collect()
    return out


def installments_features():
    print("\ninstallments_payments")
    ins = load("installments_payments")
    # positive = paid late, positive = underpaid
    ins["DAYS_LATE"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["UNDERPAID"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAY_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)

    late = (
        ins.assign(IS_LATE=(ins["DAYS_LATE"] > 0).astype("int8"))
        .groupby("SK_ID_CURR")["IS_LATE"]
        .agg(INS_LATE_N="sum", INS_LATE_RATE="mean")
    )

    out = numeric_agg(ins, "SK_ID_CURR", "INS", extra=late)
    del ins, late
    gc.collect()
    return out


def simple_table_features(name, prefix):
    print(f"\n{name}")
    df = load(name)
    out = numeric_agg(df, "SK_ID_CURR", prefix)
    del df
    gc.collect()
    return out


def main():
    print("Collapsing relational tables to applicant level\n" + "=" * 46)

    print("\napplication_train")
    app = load("application_train")
    target = app["TARGET"]
    print(f"  default rate             {target.mean():.4%}")

    # a handful of ratios that consistently carry signal in this dataset
    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    app["CREDIT_TERM"] = app["AMT_ANNUITY"] / app["AMT_CREDIT"]
    app["EMPLOYED_AGE_RATIO"] = app["DAYS_EMPLOYED"] / app["DAYS_BIRTH"]

    # 365243 is this dataset's sentinel for "not employed"
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)

    app = app.set_index("SK_ID_CURR")

    for block in (
        bureau_features(),
        previous_application_features(),
        installments_features(),
        simple_table_features("POS_CASH_balance", "POS"),
        simple_table_features("credit_card_balance", "CC"),
    ):
        app = app.join(block)
        del block
        gc.collect()

    total_rows = sum(row_counts.values())
    aux_rows = total_rows - row_counts["application_train"]

    print("\n" + "=" * 46)
    print(f"tables read              {len(row_counts)}")
    print(f"raw rows read            {total_rows:,}")
    print(f"  of which auxiliary     {aux_rows:,}")
    print(f"applicant records out    {len(app):,}")
    print(f"features out             {app.shape[1]:,}")
    print(f"compression              {total_rows / len(app):,.0f}x")

    # parquet keeps dtypes and is ~10x smaller, but needs pyarrow; fall back
    # to pickle rather than throw away an expensive aggregation
    try:
        app.to_parquet("features.parquet")
        print("\nwrote features.parquet")
    except ImportError:
        app.to_pickle("features.pkl")
        print("\npyarrow missing - wrote features.pkl instead")


if __name__ == "__main__":
    main()
