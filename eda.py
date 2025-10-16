# ---------------------------------------------------------------------
# EDA and Results Export Pipeline
# - Loads the two datasets
# - Cleans/standardises them using utils.py
# - Produces core figures and saves them to figures/
# - Runs a small set of statistical tests and exports results to JSON/TXT
# ---------------------------------------------------------------------

import os, json
import pandas as pd
import matplotlib.pyplot as plt
from utils import standardise_dataset1, standardise_dataset2
from models import chi_square, mann_whitney, kruskal


def safe_savefig(path):
    """
    Save the current matplotlib figure to 'path' with sane defaults.
    Ensures the parent directory exists, tightens layout, and closes the figure
    to avoid memory build-up when generating multiple plots.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def load_clean(root):
    """
    Load raw CSV files from the project root and apply standardisation.
    Assumptions:
      - dataset1(1).csv contains bat behaviour (risk, reward, timing, season/month)
      - dataset2(1).csv contains rat activity (arrivals, food, timing, month)
    Returns:
      d1, d2: cleaned pandas DataFrames ready for analysis/plotting.
    """
    d1 = pd.read_csv(os.path.join(root, "dataset1(1).csv"))
    d2 = pd.read_csv(os.path.join(root, "dataset2(1).csv"))
    d1 = standardise_dataset1(d1)
    d2 = standardise_dataset2(d2)
    return d1, d2


def make_figures(d1, d2, fig_dir):
    """
    Create and save key EDA figures required by the report.
    Each block checks column availability before plotting to avoid runtime errors.
    Figures are saved to the 'fig_dir' directory.
    """

    # Risk by season (expects 'season' labels like "Dry"/"Wet" and binary 'risk')
    if {"season", "risk"} <= set(d1.columns):
        ax = d1.groupby("season")["risk"].mean().plot(kind="bar")
        ax.set_title("Risk by Season (Dry vs Wet)")
        ax.set_ylabel("Mean risk rate")
        safe_savefig(os.path.join(fig_dir, "risk_by_season.png"))

    # Reward by season (expects 'season' and binary 'reward')
    if {"season", "reward"} <= set(d1.columns):
        ax = d1.groupby("season")["reward"].mean().plot(kind="bar")
        ax.set_title("Reward by Season (Dry vs Wet)")
        ax.set_ylabel("Mean reward rate")
        safe_savefig(os.path.join(fig_dir, "reward_by_season.png"))

    # Median seconds after rat arrival by risk group (robust to skew via median)
    if {"risk", "seconds_after_rat_arrival"} <= set(d1.columns):
        grouped = d1.groupby("risk")["seconds_after_rat_arrival"].median()
        ax = grouped.plot(kind="bar")
        ax.set_title("Median Seconds After Rat Arrival by Risk Group")
        ax.set_ylabel("Median seconds")
        safe_savefig(os.path.join(fig_dir, "sec_after_rat_by_risk.png"))

    # Mean hours after sunset by risk group (simple comparison of timing)
    if {"risk", "hours_after_sunset"} <= set(d1.columns):
        grouped = d1.groupby("risk")["hours_after_sunset"].mean()
        ax = grouped.plot(kind="bar")
        ax.set_title("Mean Hours After Sunset by Risk Group")
        ax.set_ylabel("Mean hours")
        safe_savefig(os.path.join(fig_dir, "hours_after_sunset_by_risk.png"))

    # Rat arrivals by month (trend across the year)
    if {"month", "rat_arrival_number"} <= set(d2.columns):
        ax = d2.groupby("month")["rat_arrival_number"].mean().plot(kind="line", marker="o")
        ax.set_title("Rat Arrivals by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Rat Arrivals")
        safe_savefig(os.path.join(fig_dir, "rat_arrivals_by_month.png"))

    # Rat arrivals as a function of hours after sunset (timing effect)
    if {"hours_after_sunset", "rat_arrival_number"} <= set(d2.columns):
        ax = d2.plot.scatter(x="hours_after_sunset", y="rat_arrival_number")
        ax.set_title("Rat Arrivals vs Hours After Sunset")
        safe_savefig(os.path.join(fig_dir, "rat_arrivals_vs_hours.png"))

    # Rat arrivals as a function of food availability (resource effect)
    if {"food_availability", "rat_arrival_number"} <= set(d2.columns):
        ax = d2.plot.scatter(x="food_availability", y="rat_arrival_number")
        ax.set_title("Rat Arrivals vs Food Availability")
        safe_savefig(os.path.join(fig_dir, "rat_arrivals_vs_food.png"))


def run_all(root):
    """
    Orchestrates the full EDA process:
      1) Load and clean data
      2) Generate figures
      3) Compute summary metrics and statistical tests
      4) Export to JSON (eda_summary.json) and TXT (quick_metrics.txt)
    """
    figures = os.path.join(root, "figures")

    # 1) Load and clean
    d1, d2 = load_clean(root)

    # 2) Figures
    make_figures(d1, d2, figures)

    # 3) Summary metrics and tests
    results = {
        "shapes": {"dataset1": list(d1.shape), "dataset2": list(d2.shape)},
        "means": {},
        "tests": {}
    }

    # Mean rates (skip if columns are absent)
    if "risk" in d1.columns:
        results["means"]["risk"] = float(d1["risk"].mean())
    if "reward" in d1.columns:
        results["means"]["reward"] = float(d1["reward"].mean())

    # Chi-square: risk vs reward (requires categorical/binary inputs)
    if {"risk", "reward"} <= set(d1.columns):
        results["tests"]["risk_vs_reward_chi2"] = chi_square(
            d1["risk"].fillna(-1),
            d1["reward"].fillna(-1)
        )

    # Mann–Whitney U: compare seconds_after_rat_arrival distribution by risk group
    if {"risk", "seconds_after_rat_arrival"} <= set(d1.columns):
        a = d1.loc[d1["risk"] == 0, "seconds_after_rat_arrival"].dropna()
        b = d1.loc[d1["risk"] == 1, "seconds_after_rat_arrival"].dropna()
        if len(a) > 3 and len(b) > 3:
            results["tests"]["sec_after_rat_by_risk_mw"] = mann_whitney(a, b)

    # Kruskal–Wallis: compare rat_arrival_number across months (non-parametric)
    if "month" in d2.columns and "rat_arrival_number" in d2.columns:
        groups = [g.dropna().values for _, g in d2.groupby(d2["month"])["rat_arrival_number"]]
        if len(groups) > 1 and all(len(g) > 2 for g in groups):
            results["tests"]["rat_arrivals_kruskal_by_month"] = kruskal(*groups)

    # 4) Export results: JSON for structured data, TXT for quick glance
    with open(os.path.join(root, "eda_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["Quick Metrics", "=============", ""]
    lines.append(f"- Dataset1 shape: {results['shapes']['dataset1']}")
    lines.append(f"- Dataset2 shape: {results['shapes']['dataset2']}")
    if "risk" in results["means"]:
        lines.append(f"- Mean risk rate: {results['means']['risk']:.3f}")
    if "reward" in results["means"]:
        lines.append(f"- Mean reward rate: {results['means']['reward']:.3f}")
    if "risk_vs_reward_chi2" in results["tests"]:
        t = results["tests"]["risk_vs_reward_chi2"]
        lines.append(f"- Risk vs Reward chi2: {t['chi2']:.2f} (p={t['p']:.3g})")

    with open(os.path.join(root, "quick_metrics.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # When the file is executed directly (python eda.py), run the pipeline
    run_all(os.path.dirname(__file__))
