# ---------------------------------------------------------------------
# models.py - Statistical Test Functions
# This script contains reusable statistical test functions used in the
# data analysis pipeline. They are imported and called by eda.py.
#
# Functions included:
# - chi_square: Tests association between two categorical variables.
# - mann_whitney: Compares two independent groups of non-normally distributed data.
# - kruskal: Compares more than two independent groups.
# ---------------------------------------------------------------------

from scipy import stats
import pandas as pd


def chi_square(series_a, series_b):
    """
    Perform a Chi-Square test of independence between two categorical variables.

    Parameters:
        series_a (pd.Series): First categorical variable.
        series_b (pd.Series): Second categorical variable.

    Returns:
        dict: Contains chi-square statistic, p-value, degrees of freedom, and the contingency table.
    """
    # Create a contingency table of observed frequencies
    ct = pd.crosstab(series_a, series_b)
    # Run the chi-square test
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # Return results as a dictionary
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "table": ct.to_dict()}


def mann_whitney(a, b):
    """
    Perform a Mann–Whitney U test to compare two independent groups.
    This is a non-parametric alternative to the two-sample t-test and does
    not assume a normal distribution.

    Parameters:
        a (array-like): Numeric values of group A.
        b (array-like): Numeric values of group B.

    Returns:
        dict: Contains U statistic, p-value, and group sizes.
    """
    # Perform the Mann–Whitney U test
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Return results as a dictionary
    return {"U": float(u), "p": float(p), "n_a": int(len(a)), "n_b": int(len(b))}


def kruskal(*groups):
    """
    Perform a Kruskal–Wallis H-test to compare more than two independent groups.
    This is a non-parametric alternative to ANOVA and is used when data is not normally distributed.

    Parameters:
        *groups: Two or more numeric groups (arrays, lists, or Series) to compare.

    Returns:
        dict: Contains H statistic and p-value.
    """
    # Perform the Kruskal–Wallis test
    H, p = stats.kruskal(*groups)
    # Return results as a dictionary
    return {"H": float(H), "p": float(p)}
