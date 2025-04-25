from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from inhibition_induced_devaluation.utils.globals import MAX_P_RESPOND, NO_RESPONSE


def fix_response_accuracy(
    df_p2: pd.DataFrame, location: str, subject_id: str
) -> pd.DataFrame:
    """
    Fix response and accuracy values for specific subjects with known data issues.

    Args:
        df_p2: DataFrame containing phase 2 data
        location: Location identifier (e.g., 'Stanford', 'UNC', 'Tel Aviv')
        subject_id: Subject identifier (e.g., 'S902', 'S4193', 'S221')

    Returns:
        pd.DataFrame: DataFrame with corrected response and accuracy values

    Notes:
        Applies specific corrections for:
        - Stanford S902 and UNC S4193: Fixes stim_location 5 responses and accuracy
        - Tel Aviv S221: Fixes stim_location 5 and 6 responses and accuracy
    """
    if (location == "Stanford" and subject_id == "S902") or (
        location == "UNC" and subject_id == "S4193"
    ):
        condition = (df_p2["stim_location"] == 5) & (df_p2["response"] == 1)
        df_p2.loc[condition, "response"] = 5
        condition2 = (
            (df_p2["stim_location"] == 5)
            & (df_p2["response"] == 5)
            & (df_p2["accuracy"] == 2)
        )
        df_p2.loc[condition2, "accuracy"] = 1
    elif location == "Tel Aviv" and subject_id == "S221":
        condition = (df_p2["stim_location"] == 5) & (df_p2["response"] == 1)
        df_p2.loc[condition, "response"] = 5
        condition2 = (
            (df_p2["stim_location"] == 5)
            & (df_p2["response"] == 5)
            & (df_p2["accuracy"] == 2)
        )
        df_p2.loc[condition2, "accuracy"] = 1
        condition3 = (df_p2["stim_location"] == 6) & (df_p2["response"] == 2)
        df_p2.loc[condition3, "response"] = 6
        condition4 = (
            (df_p2["stim_location"] == 6)
            & (df_p2["response"] == 6)
            & (df_p2["accuracy"] == 2)
        )
        df_p2.loc[condition4, "accuracy"] = 1
    return df_p2


def get_trial_types(
    df_p2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract different trial types from phase 2 data.

    Args:
        df_p2: DataFrame containing phase 2 data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        A tuple containing:
            - stop_failure_trials: Trials where stopping failed (accuracy == 3)
            - no_stop_signal_trials_go_shapes: Trials with shapes
            never paired with stopping
            - no_stop_signal_trials_stop_shapes: Go trials with
            shapes paired with stopping
            - no_stop_signal_trials_all_shapes: All go trials regardless of shape type
    """
    stop_failure_trials = df_p2.loc[df_p2["accuracy"] == 3]
    no_stop_signal_trials_go_shapes = df_p2.loc[
        df_p2["paired_with_stopping"] == 0
    ].copy()
    no_stop_signal_trials_stop_shapes = df_p2.loc[
        (df_p2["paired_with_stopping"] == 1) & (df_p2["stop_signal_trial_type"] == "go")
    ].copy()
    no_stop_signal_trials_all_shapes = df_p2.loc[
        df_p2["stop_signal_trial_type"] == "go"
    ].copy()

    return (
        stop_failure_trials,
        no_stop_signal_trials_go_shapes,
        no_stop_signal_trials_stop_shapes,
        no_stop_signal_trials_all_shapes,
    )


def calculate_mean_rts(
    stop_failure_trials: pd.DataFrame,
    no_stop_signal_trials_all_shapes: pd.DataFrame,
    no_stop_signal_trials_go_shapes: pd.DataFrame,
    no_stop_signal_trials_stop_shapes: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Calculate mean reaction times for different trial types.

    Args:
        stop_failure_trials: Trials where stopping failed
        no_stop_signal_trials_all_shapes: All go trials regardless of shape type
        no_stop_signal_trials_go_shapes: Trials with shapes never paired with stopping
        no_stop_signal_trials_stop_shapes: Go trials with shapes paired with stopping

    Returns:
        Tuple[float, float, float, float]: A tuple containing:
            - p2_go_RT: Mean RT for all go trials
            - p2_stopfail_RT: Mean RT for stop failure trials
            - p2_goRT_go_shapes: Mean RT for trials with shapes
            never paired with stopping
            - p2_goRT_stop_shapes: Mean RT for go trials with
            shapes paired with stopping
    """
    return (
        no_stop_signal_trials_all_shapes["reaction_time"].mean(),  # p2_go_RT
        stop_failure_trials["reaction_time"].mean(),  # p2_stopfail_RT
        no_stop_signal_trials_go_shapes["reaction_time"].mean(),  # p2_goRT_go_shapes
        no_stop_signal_trials_stop_shapes[
            "reaction_time"
        ].mean(),  # p2_goRT_stop_shapes
    )


def calculate_p_respond(trials_with_ss: pd.DataFrame) -> float:
    """
    Calculate probability of responding on stop signal trials.
    Returns MAX_P_RESPOND if there are no stop signal trials.
    Args:
        trials_with_SS: DataFrame containing only stop signal trials
    Returns:
        float: Probability of responding
    """
    if (trials_with_ss['response'] == NO_RESPONSE).any():
        return (
            len(trials_with_ss)
            - trials_with_ss.groupby("response").count().iloc[0].accuracy
        ) / len(trials_with_ss)
    return MAX_P_RESPOND

def calculate_nth_rt(no_stop_signal_trials_stop_shapes: pd.DataFrame,
                    rank: int) -> float:
    """
    Calculate the nth reaction time based on rank.
    Args:
        no_stop_signal_trials_stop_shapes: DataFrame of
        no-stop signal trials for stop shapes
        rank: The rank to find the nth RT
    Returns:
        float: The nth reaction time
    """
    # Replace 0 RTs with 1000
    no_stop_signal_trials_stop_shapes = no_stop_signal_trials_stop_shapes.copy()
    no_stop_signal_trials_stop_shapes["reaction_time_replaced"] = np.where(
        no_stop_signal_trials_stop_shapes["reaction_time"] == 0,
        1000,
        no_stop_signal_trials_stop_shapes["reaction_time"]
    )

    try:
        return (
            no_stop_signal_trials_stop_shapes
            .sort_values(by=["reaction_time_replaced"])
            .iloc[int(rank)]
            .reaction_time_replaced
        )
    except (IndexError, ValueError, TypeError):
        return float("nan") # return NaN if nth RT not computable


def calculate_avg_ssd(no_stop_signal_trials_stop_shapes: pd.DataFrame) -> float:
    """
    Calculate average stop signal delay.
    Args:
        no_stop_signal_trials_stop_shapes: DataFrame of
        no-stop signal trials for stop shapes
    Returns:
        float: Average stop signal delay
    """
    rank_left_trials = no_stop_signal_trials_stop_shapes.loc[
        no_stop_signal_trials_stop_shapes["stim_location"] == 5
    ]
    rank_right_trials = no_stop_signal_trials_stop_shapes.loc[
        no_stop_signal_trials_stop_shapes["stim_location"] == 6
    ]

    return (
        rank_left_trials["left_SSD"].mean() + rank_right_trials["right_SSD"].mean()
    ) / 2


def calculate_ssrt_value(nth_rt: float, avg_ssd: float) -> float:
    """
    Calculate stop signal reaction time.
    Args:
        nth_rt: The nth reaction time
        avg_ssd: Average stop signal delay
    Returns:
        float: Stop signal reaction time
    """
    if np.isnan(nth_rt):
        return float("nan") # return NaN if nth RT not computable
    return nth_rt - avg_ssd


def calculate_ssrt_components(
    df_p2: pd.DataFrame,
    no_stop_signal_trials_stop_shapes: pd.DataFrame
) -> Tuple[float, float]:
    """
    Calculate SSRT and probability of stopping.
    Args:
        df_p2: DataFrame containing phase 2 data
        no_stop_signal_trials_stop_shapes: DataFrame of no-stop signal
        trials for stop shapes
    Returns:
        Tuple[float, float]: SSRT and probability of responding
    """
    # Get trials with stop signal
    trials_with_ss = df_p2.loc[df_p2["stop_signal_trial_type"] == "stop"]

    # Calculate probability of responding
    p_respond = calculate_p_respond(trials_with_ss)

    # Calculate rank
    rank = round(p_respond * len(no_stop_signal_trials_stop_shapes))

    # Calculate components
    nth_rt = calculate_nth_rt(no_stop_signal_trials_stop_shapes, rank)
    avg_ssd = calculate_avg_ssd(no_stop_signal_trials_stop_shapes)
    ssrt = calculate_ssrt_value(nth_rt, avg_ssd)

    return ssrt, p_respond


def process_stop_signal_data(
    subject_id: str, df: pd.DataFrame, location: str
) -> Dict[str, float]:
    """
    Process stop signal data to calculate key metrics.

    Returns dictionary with keys:
    - p2_go_RT
    - p2_stopfail_RT
    - p2_goRT_go_shapes
    - p2_goRT_stop_shapes
    - p2_SSRT
    - p2_prob_stop
    """
    # Filter for part 2 only
    df_p2 = df[df["which_phase"] == "phase_2"].copy()

    # Fix response and accuracy for specific subjects
    df_p2 = fix_response_accuracy(df_p2, location, subject_id)

    # Get trial types
    (
        stop_failure_trials,
        no_stop_signal_trials_go_shapes,
        no_stop_signal_trials_stop_shapes,
        no_stop_signal_trials_all_shapes,
    ) = get_trial_types(df_p2)

    # Calculate mean RTs
    go_rt, stopfail_rt, go_shapes_rt, stop_shapes_rt = calculate_mean_rts(
        stop_failure_trials,
        no_stop_signal_trials_all_shapes,
        no_stop_signal_trials_go_shapes,
        no_stop_signal_trials_stop_shapes,
    )

    # Calculate SSRT and p(respond|signal)
    ssrt, p_respond = calculate_ssrt_components(
        df_p2, no_stop_signal_trials_stop_shapes
    )

    return {
        "p2_go_RT": go_rt,
        "p2_stopfail_RT": stopfail_rt,
        "p2_goRT_go_shapes": go_shapes_rt,
        "p2_goRT_stop_shapes": stop_shapes_rt,
        "p2_SSRT": ssrt,
        "p2_prob_stop": p_respond,
    }


def check_exclusion_criteria(
    mean_rt_stop_fail: float,
    mean_rt_no_stop_trials_stop_shapes: float,
    ssrt: float,
    min_ssrt_cutoff: float = 100,
) -> Tuple[List[int], str]:
    """Check if a subject meets exclusion criteria."""
    subject_vector = []
    # Check stop-failure RT criterion
    if mean_rt_stop_fail >= mean_rt_no_stop_trials_stop_shapes:
        subject_vector.append(0)
    else:
        subject_vector.append(1)

    # Check SSRT criterion
    if np.isnan(ssrt) or ssrt < min_ssrt_cutoff:
        subject_vector.append(0)
    else:
        subject_vector.append(1)

    # Determine exclusion reason
    if subject_vector == [1, 1]:
        reason = "include - subject passed all criteria"
    elif subject_vector == [1, 0]:
        reason = "exclude - SSRT is lower than the minimum SSRT"
    elif subject_vector == [0, 1]:
        reason = "exclude - stop fail RT is >= no-stop RT"
    else:
        reason = "exclude - failed both criteria"

    return subject_vector, reason
