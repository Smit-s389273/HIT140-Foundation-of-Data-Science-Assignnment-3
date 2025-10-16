# ---------------------------------------------------------------------
# utils.py - Data Cleaning and Standardisation Functions
# This module provides helper functions for cleaning, transforming,
# and preparing the raw datasets before analysis.
#
# Functions:
# - dry_wet_from_month: Converts a numeric month into a season label.
# - coerce_numeric: Ensures selected columns are numeric for calculations.
# - standardise_dataset1: Cleans and standardises the bat dataset.
# - standardise_dataset2: Cleans and standardises the rat dataset.
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np


def dry_wet_from_month(m):
    """
    Convert a month number (1–12) into a seasonal label ('Dry' or 'Wet').

    Parameters:
        m (int or float): Month number (1 = January, ..., 12 = December)

    Returns:
        str: "Wet" if the month is in the wet season (Nov–Apr),
             "Dry" if in the dry season (May–Oct),
             None if the month is missing or invalid.
    """
    if pd.isna(m):
        return None
    m = int(m)
    if m in [11, 12, 1, 2, 3, 4]:
        return "Wet"   # Wet season: November to April
    if m in [5, 6, 7, 8, 9, 10]:
        return "Dry"   # Dry season: May to October
    return None


def coerce_numeric(df, cols):
    """
    Convert specified DataFrame columns to numeric types.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of column names to convert.

    Returns:
        pd.DataFrame: DataFrame with converted numeric columns (invalid values become NaN).
    """
    for c in cols:
        if c in df.columns:
            # Convert the column to numeric, coercing invalid entries to NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def standardise_dataset1(df):
    """
    Clean and standardise Dataset 1 (bat behaviour data).
    - Ensures key numeric columns are converted.
    - Standardises season information into 'Dry' and 'Wet'.
    - Adds a 'season' column based on either an existing column or month mapping.

    Parameters:
        df (pd.DataFrame): Raw bat dataset.

    Returns:
        pd.DataFrame: Cleaned and standardised DataFrame.
    """
    # Convert important columns to numeric for analysis
    df = coerce_numeric(df, [
        "risk", "reward", "seconds_after_rat_arrival",
        "hours_after_sunset", "month"
    ])

    # Standardise 'season' information if it exists
    if "season" in df.columns:
        # Convert existing season encodings (0/1 or text) into Dry/Wet
        s = df["season"].astype(str).str.lower().str.strip()
        df["season_dw"] = s.map({"0": "Dry", "1": "Wet", "dry": "Dry", "wet": "Wet"})
    else:
        # If no season column, create an empty one for now
        df["season_dw"] = None

    # If 'month' column exists, infer season based on month number
    if "month" in df.columns:
        dw = df["month"].map(dry_wet_from_month)
        # Use the derived season where available; otherwise, fallback to original mapping
        df["season_dw"] = dw.where(dw.notna(), df["season_dw"])

    # Rename 'season_dw' back to 'season' and clean up
    df["season"] = df["season_dw"]
    df.drop(columns=["season_dw"], inplace=True)
    return df


def standardise_dataset2(df):
    """
    Clean and standardise Dataset 2 (rat behaviour data).
    - Ensures numeric types for key variables like timing, arrivals, and food availability.

    Parameters:
        df (pd.DataFrame): Raw rat dataset.

    Returns:
        pd.DataFrame: Cleaned and standardised DataFrame.
    """
    return coerce_numeric(df, [
        "hours_after_sunset", "bat_landing_number",
        "food_availability", "rat_minutes",
        "rat_arrival_number", "month"
    ])
