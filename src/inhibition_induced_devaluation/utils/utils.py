import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from inhibition_induced_devaluation.utils.globals import (
    EXPLICIT_KNOWLEDGE_SUBJECTS,
    MAX_P_RESPOND,
    NO_RESPONSE,
    PHASE1_EXPLICIT_KNOWLEDGE,
)

warnings.filterwarnings('ignore')

def get_project_root() -> Path:
    """Return the project root directory as a Path object."""
    return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    """Return the data directory as a Path object."""
    return get_project_root() / "data"


def get_data_locations() -> List[str]:
    """
    Get all data collection location names from the data directory.

    Returns:
        List[str]: List of directory names representing data collection locations
    """
    data_dir = get_data_dir()
    # Get only directories and filter out hidden directories (starting with .)
    return [
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]


def load_location_data(location: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a specific data collection location.

    Args:
        location (str): Name of the data collection location (e.g., 'DR2', 'UNC', etc.)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with filenames as 
        keys and pandas DataFrames as values
    """
    data_dir = get_data_dir() / location
    if not data_dir.exists():
        print(f"Warning: Location directory '{location}' does not exist")
        return {}

    csv_files = {}
    for csv_path in data_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            # Use relative path from the location directory as key
            relative_path = csv_path.relative_to(data_dir)
            csv_files[str(relative_path)] = df
        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}")

    return csv_files


def load_all_locations_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all CSV files organized by data collection location.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Dictionary with locations as keys and
                                          nested dictionaries of CSV files as values
    """
    locations = get_data_locations()
    return {location: load_location_data(location) for location in locations}


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
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - stop_failure_trials: Trials where stopping failed (accuracy == 3)
            - no_stop_signal_trials_go_shapes: Trials with shapes never paired with stopping
            - no_stop_signal_trials_stop_shapes: Go trials with shapes paired with stopping
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
            - p2_goRT_go_shapes: Mean RT for trials with shapes never paired with stopping
            - p2_goRT_stop_shapes: Mean RT for go trials with shapes paired with stopping
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
        no_stop_signal_trials_stop_shapes: DataFrame of no-stop signal trials for stop shapes
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
        return float("nan")


def calculate_avg_ssd(no_stop_signal_trials_stop_shapes: pd.DataFrame) -> float:
    """
    Calculate average stop signal delay.
    Args:
        no_stop_signal_trials_stop_shapes: DataFrame of no-stop signal trials for stop shapes
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
    if np.isnan(nth_rt) or np.isnan(avg_ssd):
        return float("nan")
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


def get_iqr_exclusions(
    data_dir: Path, excluded_subjects: Dict[str, List[str]]
) -> Dict[str, List[Dict]]:
    """
    Process all locations to identify IQR-based exclusions, only for subjects not already excluded.

    Args:
        data_dir: Path to data directory
        excluded_subjects: Dictionary of already excluded subjects by location

    Returns:
        Dict[str, List[Dict]]: Dictionary with locations as keys and lists of excluded subjects as values
    """
    exclusions = {}

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        # Get list of already excluded subjects for this location
        already_excluded = excluded_subjects.get(location_dir.name, [])

        # Process each location's data
        iid_effects = []
        subject_data = {}
        csv_files = sorted(list(location_dir.glob("*.csv")))

        # First pass: collect IID effects only for non-excluded subjects
        for csv_file in csv_files:
            subject_id = csv_file.stem
            # Skip if subject was already excluded
            if subject_id in already_excluded:
                continue

            try:
                df = pd.read_csv(csv_file)
                iid_effect, stop_bid, go_bid = process_phase3_data(df)
                if not np.isnan(iid_effect):
                    iid_effects.append(iid_effect)
                    subject_data[subject_id] = (iid_effect, stop_bid, go_bid)
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

        if not iid_effects:
            print(f"No valid IID effects found for non-excluded subjects in {location_dir.name}")
            continue

        # Calculate cutoffs and identify outliers
        upper_cutoff, lower_cutoff = calculate_iqr_cutoffs(iid_effects)
        location_exclusions = []

        for subject_id, (iid_effect, stop_bid, go_bid) in subject_data.items():
            if iid_effect > upper_cutoff or iid_effect < lower_cutoff:
                exclusion = {
                    "subject_id": subject_id,
                    "reason": "IID Effect",
                    "detailed_reason": "IID effect outside 1.5*IQR range",
                }
                location_exclusions.append(exclusion)

        if location_exclusions:
            exclusions[location_dir.name] = location_exclusions

    return exclusions

def add_explicit_knowledge_exclusions(
    location: str, behavioral_exclusions: List[Dict]
) -> List[Dict]:
    """
    Add explicit knowledge exclusions to the behavioral exclusions list.

    Args:
        location: Location identifier
        behavioral_exclusions: List of behavioral exclusion dictionaries

    Returns:
        Updated list of exclusions including explicit knowledge
    """
    # Convert behavioral exclusions to a dict for easy lookup
    excluded_subjects = {exc["subject_id"]: exc for exc in behavioral_exclusions}

    for subject in EXPLICIT_KNOWLEDGE_SUBJECTS[location]:
        if subject in excluded_subjects:
            # Update reason if subject was already excluded for behavioral reasons
            if excluded_subjects[subject]["reason"] == "Behavior":
                excluded_subjects[subject]["reason"] = "Behavior and Explicit Knowledge"
                excluded_subjects[subject]["detailed_reason"] = (
                    f"Failed behavioral criteria ({excluded_subjects[subject]['detailed_reason']}) "
                    "and reported explicit knowledge"
                )
        else:
            # Add new exclusion for explicit knowledge
            exclusion = {
                "subject_id": subject,
                "reason": "Explicit Knowledge",
                "detailed_reason": "Explicit Knowledge",
            }
            behavioral_exclusions.append(exclusion)

    return sorted(behavioral_exclusions, key=lambda x: x["subject_id"])

def get_behavioral_exclusions(file_path: Path, dataset_collection_place: str) -> Dict:
    """
    Process a single CSV file and return exclusion data.

    Args:
        file_path: Path to the subject's data file
        dataset_collection_place: Location identifier

    Returns:
        Dictionary containing exclusion data if subject should be excluded, empty dict otherwise
    """
    try:
        df = pd.read_csv(file_path)
        subject_id = file_path.stem

        # Process behavioral data
        metrics = process_stop_signal_data(subject_id, df, dataset_collection_place)
        # Check exclusion criteria
        subject_vector, reason = check_exclusion_criteria(
            metrics["p2_stopfail_RT"],
            metrics["p2_goRT_stop_shapes"],
            metrics["p2_SSRT"],
        )

        if subject_vector != [1, 1]:  # If subject should be excluded
            return {
                "subject_id": subject_id,
                "reason": "Behavior",
                "detailed_reason": reason,
                "metrics": metrics,
            }
        return {}

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}

def save_exclusion_summary(location: str, location_behavioral_reasons: Dict[str, int], location_explicit_knowledge: int, location_both: int, output_dir: Path):
    """
    Save exclusion summary for a location to a text file.

    Args:
        location: Location identifier
        location_behavioral_reasons: Dictionary mapping behavioral exclusion reasons to counts
        location_explicit_knowledge: Count of subjects excluded for explicit knowledge only
        location_both: Count of subjects excluded for both behavioral and explicit knowledge reasons
        output_dir: Path to the output directory where the file will be saved
    """
    with open(output_dir / f"{location}_exclusion_summary.txt", "w") as f:
        f.write(f"Exclusions for {location}\n")
        f.write("=" * 50 + "\n")
        f.write("\nBehavioral Exclusions:\n")
        f.write("-" * 30 + "\n")
        for reason, count in location_behavioral_reasons.items():
            if count > 0:  # Only write if there are subjects with this reason
                f.write(f"  • {reason}: {count}\n")

        f.write(f"\nExplicit Knowledge: {location_explicit_knowledge + location_both}\n")
        f.write(f"Both Behavioral and Explicit Knowledge: {location_both}\n")
        f.write(f"Total Behavioral + Explicit Knowledge Exclusions for {location}: {sum(location_behavioral_reasons.values()) + location_explicit_knowledge + location_both}\n")


def get_both_exclusions(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Process all CSV files in each location directory and return behavior + knowledge exclusion data.

    Args:
        data_dir: Path to data directory

    Returns:
        Dict[str, List[Dict]]: Dictionary with locations as keys and lists of excluded subjects as values
    """
    exclusions = {}
    output_dir = data_dir.parent / "output" / "exclusion_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        location_exclusions = []
        csv_files = sorted(list(location_dir.glob("*.csv")))

        if not csv_files:
            print(f"No CSV files found in {location_dir}")
            continue

        print(f"\nProcessing {len(csv_files)} files in {location_dir.name}")

        for csv_file in csv_files:
            exclusion_data = get_behavioral_exclusions(csv_file, location_dir.name)
            if exclusion_data:
                location_exclusions.append(exclusion_data)

        if location_exclusions:
            # Initialize location-specific counters
            location_behavioral_reasons = {}
            location_explicit_knowledge = 0
            location_both = 0

            # Count behavioral reasons before adding explicit knowledge
            for exc in location_exclusions:
                if exc["reason"] == "Behavior":
                    reason = exc["detailed_reason"]
                    location_behavioral_reasons[reason] = location_behavioral_reasons.get(reason, 0) + 1

            # Add explicit knowledge exclusions
            location_exclusions = add_explicit_knowledge_exclusions(
                location_dir.name, location_exclusions
            )

            # Count explicit knowledge and both for this location
            for exc in location_exclusions:
                if exc["reason"] == "Explicit Knowledge":
                    location_explicit_knowledge += 1
                elif exc["reason"] == "Behavior and Explicit Knowledge":
                    location_both += 1
                    # Remove from behavioral count since it's now counted as both
                    reason = exc["detailed_reason"].split(" and reported explicit knowledge")[0]
                    location_behavioral_reasons[reason] = location_behavioral_reasons.get(reason, 0) - 1
            save_exclusion_summary(location_dir.name, location_behavioral_reasons, location_explicit_knowledge, location_both, output_dir)
            exclusions[location_dir.name] = location_exclusions

    return exclusions

def process_subject_data(file_path: Path, location: str) -> pd.DataFrame:
    """
    Process individual subject data file.
    
    Args:
        file_path: Path to the subject's data file
        location: Location identifier
    Returns:
        DataFrame with processed subject data
    """
    df = pd.read_csv(file_path)
    subject_id = file_path.stem

    df["BIDDING_LEVEL"] = df["chosen_bidding_level"]
    value_level_mapping = {0.5: "L", 1: "LM", 2: "HM", 4: "H"}

    df["VALUE_LEVEL"] = df["stepwise_reward_magnitude"].map(value_level_mapping)
    df["SUBJECT"] = subject_id

    df_agg = (
        df.groupby(["SUBJECT", "VALUE_LEVEL", "paired_with_stopping"])["BIDDING_LEVEL"]
        .mean()
        .reset_index()
    )
    df_agg["STOP_CONDITION"] = df_agg["paired_with_stopping"].map(
        {0: "No Stop", 1: "Stop"}
    )

    return df_agg[["SUBJECT", "VALUE_LEVEL", "STOP_CONDITION", "BIDDING_LEVEL"]]

def calculate_mean_bids(stop_shapes: pd.DataFrame, go_shapes: pd.DataFrame) -> Tuple[float, float]:
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


def get_processed_data(
    data_dir: Path,
    excluded_subjects: Dict[str, List[str]] = None,
    subject_filter: str = "all",
) -> Dict[str, pd.DataFrame]:
    """
    Process data for all locations with different filtering options.

    Args:
        data_dir: Path to data directory
        excluded_subjects: Dictionary of excluded subjects by location
        subject_filter: One of 'all', 'included_only', or 'phase1_explicit'

    Returns:
        Dictionary with location names as keys and processed DataFrames as values
    """
    if excluded_subjects is None:
        excluded_subjects = {}

    processed_data = {}

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        location = location_dir.name
        csv_files = sorted(list(location_dir.glob("*.csv")))

        if not csv_files:
            print(f"No CSV files found in {location}")
            continue

        # Determine which subjects to process based on filter
        excluded = (
            {subj["subject_id"] for subj in excluded_subjects[location]}
            if location in excluded_subjects
            else set()
        )
        phase1_explicit = set(PHASE1_EXPLICIT_KNOWLEDGE.get(location, []))

        dfs = []
        for file_path in csv_files:
            subject_id = file_path.stem

            # Apply filters
            if subject_filter == "included_only" and subject_id in excluded:
                continue
            if (
                subject_filter == "phase1_explicit"
                and subject_id not in phase1_explicit
            ):
                continue

            try:
                df = process_subject_data(file_path, location)
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if dfs:
            processed_data[location] = pd.concat(dfs, ignore_index=True)

    return processed_data


def create_devaluation_figure(
    df: pd.DataFrame, location: str, subject_type: str, figure_dir: Path
) -> None:
    """
    Create devaluation figure from processed data.

    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        figure_dir: Path to figures directory
    """
    # Define the desired order for 'VALUE_LEVEL'
    value_level_order = pd.CategoricalDtype(["L", "LM", "HM", "H"], ordered=True)

    # Convert 'VALUE_LEVEL' to categorical type with defined order
    df["VALUE_LEVEL"] = df["VALUE_LEVEL"].astype(value_level_order)

    # Calculate statistics
    group_stats = (
        df.groupby(["VALUE_LEVEL", "STOP_CONDITION"], observed=True)["BIDDING_LEVEL"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Calculate SEM
    group_stats["SEM"] = group_stats["std"] / np.sqrt(group_stats["count"])

    # Pivot data
    avg_pivot = group_stats.pivot(
        index="VALUE_LEVEL", columns="STOP_CONDITION", values="mean"
    )
    sem_pivot = group_stats.pivot(
        index="VALUE_LEVEL", columns="STOP_CONDITION", values="SEM"
    )

    # Create figure
    plt.figure(figsize=(10, 6))
    avg_pivot.plot(kind="bar", yerr=sem_pivot, capsize=4)

    plt.title(f"{location} - {subject_type} subjects")
    plt.xlabel("Value Level")
    plt.ylabel("Average Bidding Level")
    plt.xticks(rotation=0)
    plt.yticks(np.arange(1, 7, 1))
    plt.legend(title="Stop Condition")

    # Save figure
    if location == "Stanford":
        plt.savefig(figure_dir / f"figure1a.png")
    elif location == "Tel Aviv":
        plt.savefig(figure_dir / f"figure1b.png")
    elif location == "UNC":
        plt.savefig(figure_dir / f"figure1c.png")
    elif location == "DR1":
        plt.savefig(figure_dir / f"figureS1a.png")
    elif location == "DR2":
        plt.savefig(figure_dir / f"figureS1b.png")
    plt.close()


def perform_rm_anova(
    df: pd.DataFrame, location: str, subject_type: str, output_dir: Path
) -> None:
    """
    Perform repeated measures ANOVA on the data and save results.

    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        output_dir: Path to output directory
    """
    # Ensure categorical variables
    df["STOP_CONDITION"] = df["STOP_CONDITION"].astype("category")
    df["VALUE_LEVEL"] = df["VALUE_LEVEL"].astype("category")

    # Perform repeated measures ANOVA
    anova_results = AnovaRM(
        data=df,
        depvar="BIDDING_LEVEL",
        subject="SUBJECT",
        within=["STOP_CONDITION", "VALUE_LEVEL"],
    ).fit()

    # Create anovas directory if it doesn't exist
    anova_dir = output_dir / "anovas"
    anova_dir.mkdir(parents=True, exist_ok=True)

    # Save results to file
    output_file = anova_dir / f"{location}_{subject_type}_rm_anova_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Repeated Measures ANOVA Results for {location} - {subject_type} subjects\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(anova_results))


def combine_location_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine data from all locations into a single DataFrame.

    Args:
        data_dict: Dictionary of DataFrames by location

    Returns:
        Combined DataFrame with all locations
    """
    # Add location column to each DataFrame before combining
    dfs = []
    for location, df in data_dict.items():
        df_copy = df.copy()
        df_copy["LOCATION"] = location
        dfs.append(df_copy)

    return pd.concat(dfs, ignore_index=True)


def perform_equivalence_testing(
    df: pd.DataFrame, location: str, subject_type: str, equivalence_margin: float = 0.5
) -> pd.DataFrame:
    """
    Perform equivalence testing on the data.

    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        equivalence_margin: Margin for equivalence testing

    Returns:
        DataFrame with equivalence testing results
    """
    # Ensure STOP_CONDITION is categorical
    df["STOP_CONDITION"] = df["STOP_CONDITION"].astype("category")

    # Aggregate BIDDING_LEVEL by SUBJECT and STOP_CONDITION
    aggregated_df = (
        df.groupby(["SUBJECT", "STOP_CONDITION"], observed=True)["BIDDING_LEVEL"]
        .mean()
        .reset_index()
    )

    # Pivot to align paired observations
    paired_df = aggregated_df.pivot(
        index="SUBJECT", columns="STOP_CONDITION", values="BIDDING_LEVEL"
    ).dropna()

    stop_group = paired_df["Stop"]
    no_stop_group = paired_df["No Stop"]

    # Calculate statistics
    diff = no_stop_group - stop_group
    n = len(diff)

    # Lower bound test (null: diff <= -margin)
    t_lower = (np.mean(diff) + equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)

    # Upper bound test (null: diff >= margin)
    t_upper = (np.mean(diff) - equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_upper = stats.t.cdf(t_upper, df=n - 1)

    # Create results DataFrame
    return pd.DataFrame(
        [
            {
                "Location": location,
                "Subject_Type": subject_type,
                "N": n,
                "Mean_Difference": np.mean(diff),
                "SD_Difference": np.std(diff, ddof=1),
                "Equivalence_Margin": equivalence_margin,
                "TOST_lower_t": t_lower,
                "TOST_upper_t": t_upper,
                "TOST_lower_p": p_lower,
                "TOST_upper_p": p_upper,
                "Equivalent": p_lower < 0.05 and p_upper < 0.05,
            }
        ]
    )


def convert_to_jasp_format(
    df: pd.DataFrame, location: str, subject_type: str
) -> pd.DataFrame:
    """
    Convert DataFrame to JASP format and save.

    Args:
        df: DataFrame to convert
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')

    Returns:
        Converted DataFrame in JASP format
    """
    # Aggregate: Compute mean BIDDING_LEVEL for each SUBJECT and STOP_CONDITION
    aggregated_df = (
        df.groupby(["SUBJECT", "STOP_CONDITION"])["BIDDING_LEVEL"].mean().unstack()
    )

    # Rename columns
    aggregated_df.columns = ["No_Stop", "Stop"]

    # Reset index to make SUBJECT a column
    aggregated_df.reset_index(inplace=True)

    return aggregated_df


def process_subject_files(
    data_dir: Path, location: str, excluded_subjects_list: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Process all subject files in a location and separate into
    all and included metrics."""
    location_dir = data_dir / location
    all_metrics = []
    included_metrics = []

    for file_path in location_dir.glob("*.csv"):
        subject_id = file_path.stem  # Get subject ID
        df = pd.read_csv(file_path)
        subject_metrics = process_stop_signal_data(subject_id, df, location)

        all_metrics.append(subject_metrics)

        # Correct Exclusion Check (iterate through the list of dicts)
        is_excluded = False
        for excluded_subject_dict in excluded_subjects_list:
            if excluded_subject_dict.get('subject_id') == subject_id: # safer .get to prevent errors
                is_excluded = True
                break  # No need to continue checking once found

        if not is_excluded:
            included_metrics.append(subject_metrics)

    return all_metrics, included_metrics


def calculate_metric_means(metrics_list: List[Dict], metric_key: str) -> float:
    """Calculate mean for a specific metric across subjects."""
    return np.nanmean([m[metric_key] for m in metrics_list])


def format_metric_value(value: float, is_probability: bool) -> str:
    """Format metric value according to its type."""
    return f"{value:.2f}" if is_probability else f"{int(value)}"

def create_table(metrics: Dict[str, pd.DataFrame], locs: List[str], metric_display: Dict[str, str]) -> pd.DataFrame:
    table_data = []
    for location in locs:
        row_data = {'Location': location}
        for metric_key, display_name in metric_display.items():
            mean = calculate_metric_means(metrics[location], metric_key)
            is_probability = metric_key == "p2_prob_stop"
            row_data[display_name] = format_metric_value(mean, is_probability)
        table_data.append(row_data)
    
    df = pd.DataFrame(table_data)
    return df

#Create a table with the means of metrics for each location. The location name should be in the index
def create_stopping_results_tables(data_dir: Path, table_dir: Path, excluded_subjects: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Create summary tables of stopping results comparing all vs included subjects across locations."""
    metric_display = {
        "p2_go_RT": "Go RT (ms)",
        "p2_goRT_stop_shapes": "Go RT Stop shapes (ms)",
        "p2_goRT_go_shapes": "Go RT Non-Stop Shapes (ms)",
        "p2_stopfail_RT": "Stop-Failure RT (ms)",
        "p2_prob_stop": "p(resp|signal)",
        "p2_SSRT": "SSRT (ms)",
    }

    locations = ["Stanford", "Tel Aviv", "UNC", "DR1", "DR2"]
    all_metrics = {loc: {} for loc in locations}
    included_metrics = {loc: {} for loc in locations}

    # Process all files for each location
    for location in locations:
        all_metrics[location], included_metrics[location] = process_subject_files(
            data_dir, location, excluded_subjects.get(location, [])
        )

    # Create the tables
    table1 = create_table(included_metrics, ["Stanford", "Tel Aviv", "UNC"], metric_display)
    tableS1 = create_table(all_metrics, ["Stanford", "Tel Aviv", "UNC"], metric_display)
    tableS2 = create_table(included_metrics, ["DR1", "DR2"], metric_display)
    tableS3 = create_table(all_metrics, ["DR1", "DR2"], metric_display)

    # Save the tables to the table_dir
    table1.to_csv(table_dir / "table1.csv", index=False)
    tableS1.to_csv(table_dir / "tableS1.csv", index=False)
    tableS2.to_csv(table_dir / "tableS2.csv", index=False)
    tableS3.to_csv(table_dir / "tableS3.csv", index=False)

def plot_figure2_and_s2(data, filename, title):
    plt.figure(figsize=(10, 6))

    # Determine the order of samples
    if 'DR1' in data['Sample'].unique():
        sample_order = ['DR1', 'DR2']
    else:
        sample_order = ['Stanford', 'Tel Aviv', 'UNC']

    sns.stripplot(
        data=data, x="Sample", y="Devaluation", color="gray", size=5, alpha=0.3, jitter=True,
        order=sample_order
    )

    sns.pointplot(
        data=data,
        x="Sample",
        y="Devaluation",
        join=False,
        markers="d",
        color="black",
        capsize=0.2,
        errorbar="se",
        order=sample_order
    )

    plt.xlabel("Sample")
    plt.ylabel("Devaluation")
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def create_figure2_and_s2(data_dir: Path, figure_dir: Path):
    """
    Create stripplot visualizations of IID effects by location with individual subject points and confidence intervals.
    Figure 2 includes Stanford, Tel Aviv, and UNC.
    Figure S2 includes DR1 and DR2.
    """
    # Prepare data for plotting
    plot_data = []

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        # Process each subject's data
        for csv_file in location_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                iid_effect, _, _ = process_phase3_data(df)
                if not np.isnan(iid_effect):
                    plot_data.append(
                        {"Sample": location_dir.name, "Devaluation": iid_effect}
                    )
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)

    # Create Figure 2 (Stanford, Tel Aviv, UNC)
    plot_figure2_and_s2(plot_df[plot_df['Sample'].isin(['Stanford', 'Tel Aviv', 'UNC'])], 
                  figure_dir / "figure2.png", 
                  "IID Effect with Confidence Intervals (Stanford, Tel Aviv, UNC)")

    # Create Figure S2 (DR1, DR2)
    plot_figure2_and_s2(plot_df[plot_df['Sample'].isin(['DR1', 'DR2'])], 
                  figure_dir / "figureS2.png", 
                  "IID Effect with Confidence Intervals (DR1, DR2)")


def find_devaluation_counts(positive_counts: float, negative_counts: float, zero_counts: float, iid_effect: float) -> Tuple[int, int, int]:
    """
    Update counts of positive, negative, and zero IID effects based on a new IID effect value.

    Args:
        positive_counts: Current count of positive IID effects
        negative_counts: Current count of negative IID effects
        zero_counts: Current count of zero IID effects
        iid_effect: New IID effect value to categorize

    Returns:
        Tuple[int, int, int]: Updated counts as (positive_counts, negative_counts, zero_counts)
    """
    if iid_effect > 0:
        return positive_counts + 1, negative_counts, zero_counts
    if iid_effect < 0:
        return positive_counts, negative_counts + 1, zero_counts
    return positive_counts, negative_counts, zero_counts + 1

def save_iid_effects_results_to_file(location: str, all_positive: int, all_negative: int, all_zero: int, included_positive: int, included_negative: int, included_zero: int, phase1_positive: int, phase1_negative: int, phase1_zero: int, output_dir: Path):
    """
    Save IID effect counts to a text file for a specific location.

    Args:
        location: Location identifier
        all_positive: Count of positive IID effects for all subjects
        all_negative: Count of negative IID effects for all subjects
        all_zero: Count of zero IID effects for all subjects
        included_positive: Count of positive IID effects for included subjects
        included_negative: Count of negative IID effects for included subjects
        included_zero: Count of zero IID effects for included subjects
        phase1_positive: Count of positive IID effects for phase 1 explicit knowledge subjects
        phase1_negative: Count of negative IID effects for phase 1 explicit knowledge subjects
        phase1_zero: Count of zero IID effects for phase 1 explicit knowledge subjects
        output_dir: Path to the output directory where the file will be saved
    """
    with open(output_dir / "iid_effect_counts" / f"{location}_iid_effects_counts.txt", "w") as f:
            f.write(f"Location: {location}\n")
            f.write(f"All Subjects: {all_positive + all_negative + all_zero}\n")
            f.write(f"Included Subjects: {included_positive + included_negative + included_zero}\n")
            f.write(f"Phase 1 Explicit Knowledge: {phase1_positive + phase1_negative + phase1_zero}\n")
            f.write("-" * 50 + "\n")
            f.write("All Subjects:\n")
            f.write(f"Positive IID effects: {all_positive}\n")
            f.write(f"Negative IID effects: {all_negative}\n")
            f.write(f"Zero IID effects: {all_zero}\n")
            f.write("-" * 50 + "\n")
            f.write("Included Subjects:\n")
            f.write(f"Positive IID effects: {included_positive}\n")
            f.write(f"Negative IID effects: {included_negative}\n")
            f.write(f"Zero IID effects: {included_zero}\n")
            f.write("-" * 50 + "\n")
            if location in PHASE1_EXPLICIT_KNOWLEDGE:
                f.write("Phase 1 Explicit Knowledge:\n")
                f.write(f"Positive IID effects: {phase1_positive}\n")
                f.write(f"Negative IID effects: {phase1_negative}\n")
                f.write(f"Zero IID effects: {phase1_zero}\n")
                f.write("-" * 50 + "\n")

def analyze_iid_effects_by_site(data_dir: Path, excluded_subjects: Dict[str, List[Dict]] = None, output_dir: Path = None):
    """
    Analyze IID effects by site, counting subjects with positive and negative effects.
    Prints results for both all subjects and included subjects.

    Args:
        data_dir: Path to data directory
        excluded_subjects: Dictionary of excluded subjects by location
    """
    if excluded_subjects is None:
        excluded_subjects = {}

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        location = location_dir.name
        excluded_ids = {subj["subject_id"] for subj in excluded_subjects.get(location, [])}

        # Initialize counters for all subjects
        all_positive, all_negative, all_zero = 0, 0, 0
        # Initialize counters for included subjects
        included_positive, included_negative, included_zero = 0, 0, 0

        # Initialize counters for phase 1 explicit knowledge subjects
        phase1_positive, phase1_negative, phase1_zero = 0, 0, 0

        # Process each subject's data
        for csv_file in location_dir.glob("*.csv"):
            subject_id = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                iid_effect, _, _ = process_phase3_data(df)

                if not np.isnan(iid_effect):
                    # Count for all subjects
                    all_positive, all_negative, all_zero = find_devaluation_counts(all_positive, all_negative, all_zero, iid_effect)

                    # Count for included subjects
                    if subject_id not in excluded_ids:
                        included_positive, included_negative, included_zero = find_devaluation_counts(included_positive, included_negative, included_zero, iid_effect)

                    if location in PHASE1_EXPLICIT_KNOWLEDGE:
                        if subject_id in PHASE1_EXPLICIT_KNOWLEDGE[location]:
                            phase1_positive, phase1_negative, phase1_zero = find_devaluation_counts(phase1_positive, phase1_negative, phase1_zero, iid_effect)

            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

        save_iid_effects_results_to_file(location, all_positive, all_negative, all_zero, included_positive, included_negative, included_zero, phase1_positive, phase1_negative, phase1_zero, output_dir)

def run_standard_analyses(
    df: pd.DataFrame,
    location: str,
    subject_type: str,
    figure_dir: Path,
    output_dir: Path,
    jasp_dir: Path,
) -> pd.DataFrame:
    """Runs the standard set of analyses and saves outputs for a given dataset.

    Args:
        df: DataFrame containing the data to analyze.
        location: Location identifier (e.g., 'UNC', 'combined').
        subject_type: Type of subjects ('all', 'included', 'phase1').
        figure_dir: Path to the directory for saving figures.
        output_dir: Path to the directory for saving ANOVA/Equivalence results.
        jasp_dir: Path to the directory for saving JASP files.

    Returns:
        DataFrame containing the results of the equivalence test.
    """
    # Create devaluation figure
    if subject_type == "included" and location != "combined":
        create_devaluation_figure(df, location, subject_type, figure_dir)

    # Perform RM ANOVA
    perform_rm_anova(df, location, subject_type, output_dir)

    # Perform equivalence testing
    equiv_results = perform_equivalence_testing(df, location, subject_type)

    # Convert to JASP format and save
    jasp_df = convert_to_jasp_format(df, location, subject_type)
    jasp_df.to_csv(jasp_dir / f"{location}_{subject_type}_jasp.csv", index=False)

    return equiv_results