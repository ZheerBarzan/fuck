#!/usr/bin/env python3
# train_model.py  (correlations v3)

"""
Usage
─────────────────
⭕ python train_model.py
⭕ python train_model.py --model rf
⭕ python train_model.py --scatter "Land area" "Sale price of the property in U$"
⭕ python train_model.py --cor "Age of the property in years"

"""
import argparse
import sys
import warnings
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")          # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycaret.regression import (
    compare_models, create_model, finalize_model,
    save_model, setup, tune_model, pull
)
import matplotlib.ticker as mticker

# ─────────────────────── Config ───────────────────────
TARGET        = "Sale price of the property in U$"
DEF_XLSX      = "erbil_data.xlsx"
CORR_IMG      = "./results/correlation_heatmap.png"
CORR_CSV      = "./results/feature_correlations.csv"
MODEL_OUT     = "best_price_model"
METRICS_CSV   = "./results/model_metrics.csv"

# ─────────────────────── CLI ───────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fast + safe PyCaret trainer")
    p.add_argument("-e", "--excel", type=Path, default=DEF_XLSX)
    p.add_argument("-m", "--model", type=str, default=None,
                   help="rf | ridge | lgbm … (blank = auto-select)")
    # pruning (ignored if --keep-all)
    p.add_argument("--numeric-only", action="store_true",
                   help="ignore all categoricals")
    p.add_argument("--cat-max", type=int, default=30,
                   help="drop categoricals with > N unique values")
    p.add_argument("--feature-top", type=int, default=200,
                   help="keep only K numerics most correlated with target")
    # correlation output
    p.add_argument("--skip-corr", action="store_true",
                   help="skip heat-map + CSV export")
    p.add_argument("--corr-top", type=int, default=30,
                   help="heat-map size — 0 = all numeric columns")
    # NEW
    p.add_argument("--keep-all", action="store_true",
                   help="disable pruning, ingest full sheet")
    # Scatter plot for given columns
    p.add_argument("--scatter", nargs='+', metavar='COLUMN',
                   help="Generate scatter plot for given columns (2 required)")
    # Correlation with target
    p.add_argument("--cor", metavar='COLUMN',
                   help="Show correlation and plot with target for given column")
    return p.parse_args()

# ─────────────────────── Helpers ───────────────────────
def flatten_multiindex(cols: pd.MultiIndex) -> list[str]:
    return [
        " | ".join(
            str(lvl).strip() for lvl in tup
            if str(lvl).strip() and str(lvl).lower() != "nan"
        )
        for tup in cols
    ]

def to_numeric_for_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy where *every* column is numeric (factorise categoricals)."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            # factorize: unseen / NaN become -1
            out[col] = pd.factorize(out[col], sort=True)[0]
    return out

def sanitize_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s)

# ─────────────────────── Data wrangling ───────────────────────
def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[!] File not found: {path}")

    try:
        df = pd.read_excel(path, engine="openpyxl")
    except ValueError:
        df = pd.read_excel(path, engine="openpyxl", header=[0, 1, 2])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = flatten_multiindex(df.columns)

    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")

    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    df[TARGET] = df[TARGET].astype(float)
    return df

def reduce_features(df: pd.DataFrame, args) -> tuple[pd.DataFrame, list[str]]:
    if args.keep_all:
        print("[↷] --keep-all → no features pruned")
        return df, []

    ignore: list[str] = []

    if args.numeric_only:
        ignore = df.select_dtypes(exclude=[np.number]).columns.tolist()
        df = df.drop(columns=ignore)
        print(f"[↷] numeric-only → {df.shape[1]} numeric features kept")
        return df, ignore

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique(dropna=False) > args.cat_max:
            ignore.append(col)
        else:
            df[col] = df[col].astype(str)
    df = df.drop(columns=ignore)

    num = df.select_dtypes(include=[np.number])
    best = (
        num.corr()[TARGET].abs()
        .sort_values(ascending=False)
        .head(args.feature_top)
        .index
    )
    drop_extra = [c for c in num.columns if c not in best and c != TARGET]
    if drop_extra:
        ignore.extend(drop_extra)
        df = df.drop(columns=drop_extra)

    print(f"[↷] after pruning: {df.shape[1]} features "
          f"({len(ignore)} columns ignored)")
    return df, ignore

# ─────────────────────── Correlations ───────────────────────
def save_correlations_csv(df: pd.DataFrame):
    num_all = to_numeric_for_corr(df)
    corr_series = (
        num_all.corr()[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )
    corr_series.to_csv(CORR_CSV, header=["Correlation"])
    print(f"[✓] full correlation table → {CORR_CSV}")

def heatmap(df: pd.DataFrame, top_n: int):
    num_all = to_numeric_for_corr(df)
    corr     = num_all.corr()

    if top_n <= 0 or top_n >= len(num_all.columns):
        cols  = corr.columns
        title = "Correlation heat-map for **all** features"
    else:
        cols  = corr[TARGET].abs().sort_values(ascending=False).head(top_n).index
        title = f"Top {len(cols)} correlations with {TARGET}"

    corr = corr.loc[cols, cols]

    fig_w = max(10, 0.5 * len(cols))
    fig_h = max(8,  0.4 * len(cols))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr, cmap="coolwarm",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=.4, cbar_kws={"shrink": .8},
        annot=True, fmt=".2f", annot_kws={"size": 8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(CORR_IMG, dpi=300)
    plt.close()
    print(f"[✓] {CORR_IMG} saved")

# ─────────────────────── Training ───────────────────────
def train(df: pd.DataFrame, ignore, model_id):
    warnings.filterwarnings("ignore", category=UserWarning)

    setup(
        data=df,
        target=TARGET,
        ignore_features=ignore,
        html=False,
        session_id=42,
        n_jobs=-1,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
        verbose=False,
    )


    from pycaret.regression import get_config

    # Log and optionally save the train/test split
    X_train = get_config("X_train")
    X_test = get_config("X_test")
    print(f"[i] Train/Test Split → {X_train.shape[0]} training rows, {X_test.shape[0]} test rows")

    X_train.to_csv("./results/train_data.csv", index=False)
    X_test.to_csv("./results/test_data.csv", index=False)

    if model_id:
        mdl = tune_model(
            create_model(model_id, fold=5),
            choose_better=True, fold=5
        )
    else:
        mdl = compare_models(sort="RMSE", turbo=True)

    pull().to_csv(METRICS_CSV, index=False)
    print(f"[✓] metrics saved → {METRICS_CSV}")

    save_model(finalize_model(mdl), MODEL_OUT)
    print(f"[✓] model stored → {MODEL_OUT}.pkl")

def scatter_plot(df: pd.DataFrame, cols: list[str]):
    """Generate and save a scatter plot for the given columns, handling categorical columns and excluding outliers (1st-99th percentile for numerics)."""
    if len(cols) < 2:
        print("[!] --scatter requires at least 2 column names.")
        return
    x, y = cols[0], cols[1]
    if x not in df.columns or y not in df.columns:
        print(f"[!] Columns not found: {x}, {y}")
        return
    x_is_cat = not pd.api.types.is_numeric_dtype(df[x])
    y_is_cat = not pd.api.types.is_numeric_dtype(df[y])
    if x_is_cat and y_is_cat:
        print(f"[!] Both {x} and {y} are categorical. Scatter plot is not meaningful.")
        return
    plt.figure(figsize=(10, 6))
    if x_is_cat or y_is_cat:
        # One is categorical, one is numeric
        if x_is_cat:
            sns.stripplot(x=df[x], y=df[y], alpha=0.6, jitter=True)
            plt.xlabel(x)
            plt.ylabel(y)
        else:
            sns.stripplot(x=df[y], y=df[x], alpha=0.6, jitter=True)
            plt.xlabel(y)
            plt.ylabel(x)
        plt.title(f"Strip plot: {x} vs {y}")
    else:
        # Both numeric: remove outliers
        x_vals = df[x]
        y_vals = df[y]
        x_low, x_high = x_vals.quantile(0.01), x_vals.quantile(0.99)
        y_low, y_high = y_vals.quantile(0.01), y_vals.quantile(0.99)
        mask = (x_vals >= x_low) & (x_vals <= x_high) & (y_vals >= y_low) & (y_vals <= y_high)
        filtered = df[mask]
        n_excluded = len(df) - len(filtered)
        if n_excluded > 0:
            print(f"[i] Excluded {n_excluded} outlier points for clearer visualization.")
        plt.scatter(filtered[x], filtered[y], alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Scatter plot: {x} vs {y}")
        # Format y-axis as plain numbers
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        # Add diagonal line (y = x) for reference
        min_val = max(filtered[x].min(), filtered[y].min())
        max_val = min(filtered[x].max(), filtered[y].max())
        if max_val > min_val:
            plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=2, label='y = x')
            plt.legend()
    out_path = f"./results/scatter_{sanitize_filename(x)}_vs_{sanitize_filename(y)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Scatter/strip plot saved → {out_path}")

def correlation_with_target(df: pd.DataFrame, col: str):
    """Show correlation and plot with target for given column, handling categoricals and excluding outliers (1st-99th percentile for numerics)."""
    if col not in df.columns:
        print(f"[!] Column not found: {col}")
        return
    x = df[col]
    y = df[TARGET]
    # Drop rows with NaN in either column
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    x_is_cat = not pd.api.types.is_numeric_dtype(x)
    if x_is_cat:
        # Factorize for correlation
        x_codes, uniques = pd.factorize(x, sort=True)
        corr = np.corrcoef(x_codes, y)[0, 1]
        print(f"[✓] Correlation (factorized) between '{col}' and '{TARGET}': {corr:.4f}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y)
        plt.xlabel(col)
        plt.ylabel(TARGET)
        plt.title(f"Box plot: {col} vs {TARGET}\nPearson r (factorized) = {corr:.3f}")
        # Format y-axis as plain numbers
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        # Remove outliers (1st-99th percentile)
        x_low, x_high = x.quantile(0.01), x.quantile(0.99)
        y_low, y_high = y.quantile(0.01), y.quantile(0.99)
        mask2 = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
        x = x[mask2]
        y = y[mask2]
        n_excluded = mask.size - mask2.sum()
        if n_excluded > 0:
            print(f"[i] Excluded {n_excluded} outlier points for clearer visualization.")
        if x.empty or y.empty:
            print(f"[!] No data for correlation between {col} and {TARGET}")
            return
        corr = np.corrcoef(x, y)[0, 1]
        print(f"[✓] Correlation between '{col}' and '{TARGET}': {corr:.4f}")
        plt.figure(figsize=(8, 6))
        sns.regplot(x=x, y=y, scatter_kws={'alpha':0.6})
        plt.xlabel(col)
        plt.ylabel(TARGET)
        plt.title(f"Correlation: {col} vs {TARGET}\nPearson r = {corr:.3f}")
        # Format y-axis as plain numbers
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    out_path = f"./results/correlation_{sanitize_filename(col)}_vs_target.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Correlation plot saved → {out_path}")



# ─────────────────────── main ───────────────────────
def main():
    args = cli()
    print(f"[•] reading {args.excel} …")
    df = load_excel(args.excel)
    print(f"[✓] {len(df):,} rows × {df.shape[1]} raw columns")

    # Scatter plot option
    if args.scatter:
        scatter_plot(df, args.scatter)
        return
    # Correlation with target option
    if args.cor:
        correlation_with_target(df, args.cor)
        return

    df, ignore = reduce_features(df, args)

    if not args.skip_corr:
        save_correlations_csv(df)
        heatmap(df, args.corr_top)

    train(df, ignore, args.model)

if __name__ == "__main__":
    main()
