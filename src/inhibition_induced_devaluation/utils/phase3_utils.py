from typing import List, Tuple

import numpy as np
import pandas as pd


def calculate_mean_bids(stop_shapes: pd.DataFrame,
                        go_shapes: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate mean bidding levels for stop and go shapes.

    Args:
        stop_shapes: DataFrame containing data for shapes paired with stopping
        go_shapes: DataFrame containing data for shapes not paired with stopping

    Returns:
        Tuple[float, float]: A tuple containing (stop_bid, go_bid) where:
            - stop_bid: Mean bidding level for shapes paired with stopping
            - go_bid: Mean bidding level for shapes not paired with stopping
    """
    stop_bid = stop_shapes["chosen_bidding_level"].mean()
    go_bid = go_shapes["chosen_bidding_level"].mean()
    return stop_bid, go_bid

# Split process_phase3_data into smaller functions
def get_phase3_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract phase 3 data from DataFrame."""
    return df[df["which_phase"] == "phase_3"].copy()

def separate_shape_types(df_p3: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate stop and go shapes."""
    stop_shapes = df_p3[df_p3["paired_with_stopping"] == 1]
    go_shapes = df_p3[df_p3["paired_with_stopping"] == 0]
    return stop_shapes, go_shapes

def calculate_iid_effect(stop_bid: float, go_bid: float) -> float:
    """Calculate IID effect."""
    return go_bid - stop_bid  # Positive values indicate devaluation

def process_phase3_data(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Process phase 3 data to calculate IID effects.

    Args:
        df: DataFrame containing phase 3 data
    Returns:
        Tuple containing (iid_effect, stop_bid, go_bid)
    """
    df_p3 = df[df["which_phase"] == "phase_3"].copy()

    stop_shapes, go_shapes = separate_shape_types(df_p3)
    stop_bid, go_bid = calculate_mean_bids(stop_shapes, go_shapes)
    iid_effect = calculate_iid_effect(stop_bid, go_bid)

    return iid_effect, stop_bid, go_bid


def calculate_iqr_cutoffs(iid_effects: List[float]) -> Tuple[float, float]:
    """
    Calculate IQR-based cutoffs for outlier detection.

    Args:
        iid_effects: List of IID effect values
    Returns:
        Tuple containing (upper_cutoff, lower_cutoff)
    """
    q75, q25 = np.percentile(iid_effects, [75, 25])
    iqr = q75 - q25
    upper_cutoff = q75 + iqr * 1.5
    lower_cutoff = q25 - iqr * 1.5
    return upper_cutoff, lower_cutoff
