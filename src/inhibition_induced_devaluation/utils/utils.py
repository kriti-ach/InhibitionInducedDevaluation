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
    BAYES_FACTORS,
    EXPLICIT_KNOWLEDGE_SUBJECTS,
    PHASE1_EXPLICIT_KNOWLEDGE,
)
from inhibition_induced_devaluation.utils.phase2_utils import (
    check_exclusion_criteria,
    process_stop_signal_data,
)
from inhibition_induced_devaluation.utils.phase3_utils import (
    calculate_iqr_cutoffs,
    process_phase3_data,
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


def get_iqr_exclusions(
    data_dir: Path, excluded_subjects: Dict[str, List[str]]
) -> Dict[str, List[Dict]]:
    """
    Process all locations to identify IQR-based exclusions,
    only for subjects not already excluded.

    Args:
        data_dir: Path to data directory
        excluded_subjects: Dictionary of already excluded subjects by location

    Returns:
        Dict[str, List[Dict]]: Dictionary with locations as
        keys and lists of excluded subjects as values
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
            print("No valid IID effects found for non-excluded" +
                  f"subjects in {location_dir.name}")
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
                    f"Failed behavioral criteria ({excluded_subjects[subject]
                                                   ['detailed_reason']}) "
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
        Dictionary containing exclusion data if subject
        should be excluded, empty dict otherwise
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

def save_exclusion_summary(location: str, location_behavioral_reasons:
                           Dict[str, int], location_explicit_knowledge: int,
                           location_both: int, output_dir: Path):
    """
    Save exclusion summary for a location to a text file.

    Args:
        location: Location identifier
        location_behavioral_reasons: Dictionary mapping behavioral
        exclusion reasons to counts
        location_explicit_knowledge: Count of subjects excluded for
        explicit knowledge only
        location_both: Count of subjects excluded for both behavioral
        and explicit knowledge reasons
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

        f.write(
            f"\nExplicit Knowledge: "
            f"{location_explicit_knowledge + location_both}\n"
        )
        f.write(
            f"Both Behavioral and Explicit Knowledge: "
            f"{location_both}\n"
        )
        total_exclusions = (sum(location_behavioral_reasons.values()) +
                    location_explicit_knowledge + location_both)
        f.write(
            f"Total Behavioral + Explicit Knowledge Exclusions for {location}: "
            f"{total_exclusions}\n"
        )

def get_both_exclusions(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Process all CSV files in each location directory and
    return behavior + knowledge exclusion data.

    Args:
        data_dir: Path to data directory

    Returns:
        Dict[str, List[Dict]]: Dictionary with locations as
        keys and lists of excluded subjects as values
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
                    location_behavioral_reasons[reason] = (
                        location_behavioral_reasons.get(reason, 0) + 1
                        )

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
                    reason = exc["detailed_reason"].split
                    (" and reported explicit knowledge")[0]
                    location_behavioral_reasons[reason] = (
                        location_behavioral_reasons.get(reason, 0) - 1
                        )
            save_exclusion_summary(location_dir.name,
                                   location_behavioral_reasons,
                                   location_explicit_knowledge,
                                   location_both, output_dir)
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
        {0: "No stop", 1: "Stop"}
    )

    return df_agg[["SUBJECT", "VALUE_LEVEL", "STOP_CONDITION", "BIDDING_LEVEL"]]

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
    df: pd.DataFrame,
    location: str,
    subject_type: str,
    figure_dir: Path,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create devaluation figure from processed data.

    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        figure_dir: Path to figures directory
        ax: Optional matplotlib Axes object for subplotting
    Returns:
        matplotlib Axes object
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
    color_map = {"No stop": "#444444", "Stop": "#ffffff"}
    bar_colors = [color_map.get(col, "#cccccc") for col in avg_pivot.columns]
    avg_pivot.plot(kind="bar", yerr=sem_pivot, capsize=4, ax=ax, color=bar_colors, edgecolor='black')

    if subject_type == "included":
        ax.set_title(f"{location}", fontsize=20)
    elif subject_type == "all":
        ax.set_title(f"{location} - No Exclusions", fontsize=20)
    else:
        ax.set_title(f"{location} - Phase 1 Explicit Learners", fontsize=20)
    ax.set_xlabel("Value Level", fontsize=16)
    ax.set_ylabel("Average Bidding Level", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14)
    ax.set_yticks(np.arange(1, 7, 1))
    ax.tick_params(axis='y', labelsize=14)
    leg = ax.legend(title="Stop Condition", fontsize=14, title_fontsize=14)
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(14)
        leg.get_title().set_fontsize(14)

    return ax  # Return the axes object, it can be used if no ax provided.

def create_combined_devaluation_figures(data,
                                        figure_dir: Path,
                                        subject_type: str,
                                        include_dr: bool = False,
                                        main_figure_name: str = "",
                                        dr_figure_name: str = ""):
    """Creates combined devaluation figures (side-by-side plots)."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    locations_abc = ["Stanford", "Tel Aviv", "UNC"]

    for i, location in enumerate(locations_abc):
        df = data[location]
        create_devaluation_figure(df, location, subject_type, figure_dir, axes[i])
    plt.tight_layout()
    plt.savefig(figure_dir / main_figure_name)
    plt.close(fig)  # Close the figure to prevent memory issues.

    if include_dr:
        # Create the combined figure for DR1 and DR2
        fig_s, axes_s = plt.subplots(1, 2, figsize=(12, 6))
        locations_s = ["DR1", "DR2"]
        for i, location in enumerate(locations_s):
            df = data[location]
            create_devaluation_figure(df, location, subject_type, figure_dir, axes_s[i])
        plt.tight_layout()
        plt.savefig(figure_dir / dr_figure_name)
        plt.close(fig_s)

def perform_rm_anova(
    df: pd.DataFrame) -> AnovaRM:
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
    return AnovaRM(
        data=df,
        depvar="BIDDING_LEVEL",
        subject="SUBJECT",
        within=["STOP_CONDITION", "VALUE_LEVEL"],
    ).fit()

def combine_location_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine data from all 3 locations into a single DataFrame.

    Args:
        data_dict: Dictionary of DataFrames by location

    Returns:
        Combined DataFrame with all locations
    """
    # Add location column to each DataFrame before combining
    dfs = []
    for location, df in data_dict.items():
        if location != "DR1" and location != "DR2":
            df_copy = df.copy()
            df_copy["LOCATION"] = location
            dfs.append(df_copy)

    return pd.concat(dfs, ignore_index=True)


def perform_equivalence_testing(
    df: pd.DataFrame, location: str, subject_type: str, equivalence_margin: float = 0.24
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
    no_stop_group = paired_df["No stop"]

    # Calculate statistics
    diff = no_stop_group - stop_group
    n = len(diff)

    # Lower bound test (null: diff <= -margin)
    t_lower = (np.mean(diff) + equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)

    # Upper bound test (null: diff >= margin)
    t_upper = (np.mean(diff) - equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    return t_lower, t_upper, p_lower, p_upper


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
            if excluded_subject_dict.get('subject_id') == subject_id:
                is_excluded = True
                break  # No need to continue checking once found

        if not is_excluded:
            included_metrics.append(subject_metrics)

    return all_metrics, included_metrics


def calculate_metric_means(metrics_list: List[Dict],
                           metric_key: str) -> Tuple[float, float]:
    """Calculate mean and standard deviation for a specific metric across subjects."""
    values = [m[metric_key] for m in metrics_list]
    return np.nanmean(values), np.nanstd(values)  # nanmean and nanstd to ignore NaNs


def format_metric_value(mean: float, std: float, is_probability: bool) -> str:
    """Format metric value with standard deviation according to its type."""
    if is_probability:
        return f"{mean:.2f} ({std:.2f})"
    return f"{int(mean)} ({int(std)})"

def populate_stopping_results_table(metrics: Dict[str, pd.DataFrame],
                                    locs: List[str], metric_display:
                                    Dict[str, str]) -> pd.DataFrame:
    table_data = []
    for location in locs:
        row_data = {'Location': location}
        for metric_key, display_name in metric_display.items():
            mean, std = calculate_metric_means(metrics[location], metric_key)
            is_probability = metric_key == "p2_prob_stop"
            row_data[display_name] = format_metric_value(mean, std, is_probability)
        table_data.append(row_data)

    return pd.DataFrame(table_data)

# Create a table with the means of metrics for each location.
# The location name should be in the index
def create_stopping_results_tables(data_dir: Path, table_dir: Path,
                                   excluded_subjects: Dict[str, List[str]]) -> Dict[
                                       str, pd.DataFrame]:
    """Create summary tables of stopping results comparing all
    vs included subjects across locations."""
    metric_display = {
        "p2_go_RT": "Go RT (ms)",
        "p2_goRT_stop_shapes": "Go RT Stop shapes (ms)",
        "p2_goRT_go_shapes": "Go RT No stop Shapes (ms)",
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
    table1 = populate_stopping_results_table(included_metrics,
                                             ["Stanford", "Tel Aviv", "UNC"],
                                             metric_display)
    tables1 = populate_stopping_results_table(all_metrics,
                                              ["Stanford", "Tel Aviv", "UNC"],
                                              metric_display)
    tables2 = populate_stopping_results_table(included_metrics,
                                              ["DR1", "DR2"],
                                              metric_display)
    tables3 = populate_stopping_results_table(all_metrics,
                                              ["DR1", "DR2"],
                                              metric_display)

    # Save the tables to the table_dir
    table1.to_csv(table_dir / "table1.csv", index=False)
    tables1.to_csv(table_dir / "tableS1.csv", index=False)
    tables2.to_csv(table_dir / "tableS4.csv", index=False)
    tables3.to_csv(table_dir / "tableS5.csv", index=False)

def plot_figure2(data, filename, ylim: Tuple[float, float] = (-4, 4)):
    plt.figure(figsize=(10, 6))

    # Determine the order of samples
    if 'DR1' in data['Sample'].unique():
        sample_order = ['DR1', 'DR2']
    else:
        sample_order = ['Stanford', 'Tel Aviv', 'UNC']

    sns.stripplot(
        data=data, x="Sample", y="Devaluation", color="gray",
        size=5, alpha=0.3, jitter=True,
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

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Devaluation", fontsize=16)
    plt.ylim(ylim)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    # If a legend is present, set its font size
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(14)
        leg.get_title().set_fontsize(14)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def create_figure2(data_dir: Path, figure_dir: Path):
    """
    Create stripplot visualizations of IID effects by location
    with individual subject points and confidence intervals.
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
    plot_figure2(plot_df[plot_df['Sample'].isin(['Stanford',
                                                        'Tel Aviv', 'UNC'])],
                  figure_dir / "figure2.png")


def find_devaluation_counts(positive_counts: float, negative_counts: float,
                            zero_counts: float, iid_effect: float) -> Tuple[
                                int, int, int]:
    """
    Update counts of positive, negative, and zero IID effects based on a
    new IID effect value.

    Args:
        positive_counts: Current count of positive IID effects
        negative_counts: Current count of negative IID effects
        zero_counts: Current count of zero IID effects
        iid_effect: New IID effect value to categorize

    Returns:
        Tuple[int, int, int]: Updated counts as
        (positive_counts, negative_counts, zero_counts)
    """
    if iid_effect > 0:
        return positive_counts + 1, negative_counts, zero_counts
    if iid_effect < 0:
        return positive_counts, negative_counts + 1, zero_counts
    return positive_counts, negative_counts, zero_counts + 1

def format_f_statistics(stats, name):
    """
    Format F-statistic results into a string.

    Args:
        stats (dict): A dictionary containing the statistical results with keys:
                      'Num DF', 'Den DF', 'F Value', 'Pr > F'.
        name (str): Name of the statistical test
        (e.g., 'Stopping', 'Value', 'Interaction').

    Returns:
        str: Formatted string with F-statistic results.
    """
    return (f"{name} F({int(stats['Num DF'])}, "
            f"{int(stats['Den DF'])}) = "
            f"{stats['F Value']:.2f}, p "
            f"{'< .001' if stats['Pr > F'] < 0.001 else f'= {stats['Pr > F']:.3f}'}")

def create_results_table(
    data_dict: Dict[str, pd.DataFrame],
    subject_type: str,
    output_dir: Path,
    locations: List[str] = ["Stanford", "Tel Aviv", "UNC", "Combined"],
    is_dr_site: bool = False
) -> pd.DataFrame:
    # Initialize a DataFrame to store results with a multi-level index
    index = pd.MultiIndex.from_tuples([
        ('Stopping', 'Main effect'),
        ('Stopping', 'BF₀₁'),
        ('Stopping', 'Equivalence test'),
        ('Stopping', 'Count of No stop <= Stop'),
        ('Value', 'Main effect'),
        ('Interaction', 'Interaction'),
    ], names=['Effect', 'Test'])
    results = pd.DataFrame(index=index)

    # Process each location
    for location in locations:
        if location not in data_dict:
            print(f"No data available for {location}")
            continue
        if location == "Combined":
            if is_dr_site:
                continue
        df = data_dict[location]

        # ANOVA and other calculations
        anova_result = perform_rm_anova(df)

        # Get F-values, degrees of freedom, and p-values for each effect
        stopping_stats = anova_result.anova_table.loc["STOP_CONDITION"]
        value_stats = anova_result.anova_table.loc["VALUE_LEVEL"]
        interaction_stats = anova_result.anova_table.loc["STOP_CONDITION:VALUE_LEVEL"]

        # Format ANOVA results
        stopping_string = format_f_statistics(stopping_stats, 'Stopping')
        value_string = format_f_statistics(value_stats, 'Value')
        interaction_string = format_f_statistics(interaction_stats, 'Interaction')

        # Equivalence testing
        t_lower, t_upper, p_lower, p_upper = perform_equivalence_testing(
            df, location, subject_type)
        equiv_p = max(p_lower, p_upper)
        equiv_t = t_lower if abs(t_lower) < abs(t_upper) else t_upper

        # Count subjects where No stop <= Stop
        aggregated_df = df.groupby(["SUBJECT", "STOP_CONDITION"],
                                   observed=True)["BIDDING_LEVEL"].mean().reset_index()
        paired_df = aggregated_df.pivot(index="SUBJECT",
                                        columns="STOP_CONDITION",
                                        values="BIDDING_LEVEL").dropna()

        nostop_less_than_equal_to_stop = sum(paired_df["No stop"] <= paired_df["Stop"])
        total_subjects = len(paired_df)
        n = len(paired_df)

        # Get Bayes Factor
        bf_key = f"{subject_type.lower()}_{location.lower().replace(' ', '_')}"
        bf_value = BAYES_FACTORS.get(bf_key, "")
        bf_string = f"{bf_value:.2f}" if bf_value != "" else ""

        # Store results for this location
        results[location] = [
            stopping_string,
            bf_string,
            f"t({n-1}) = {equiv_t:.2f}, p {'< .001' if
                                           equiv_p < 0.001 else f'= {equiv_p:.3f}'}",
            f"{nostop_less_than_equal_to_stop} of {total_subjects}",
            value_string,
            interaction_string,
        ]

    # Determine the output file
    if is_dr_site:
        output_file = output_dir / ("tableS6.csv" if
                                    subject_type == "included" else "tableS7.csv")
    else:
        if subject_type == "included":
            output_file = output_dir / "table2.csv"
        elif subject_type == "all":
            output_file = output_dir / "tableS2.csv"
        elif subject_type == "phase1":
            output_file = output_dir / "tableS3.csv"

    results.to_csv(output_file)
    return results

def add_combined_data(data_dict: Dict[str, pd.DataFrame],
                      combined_data: pd.DataFrame) -> None:
    """
    Add combined data to the data dictionary if provided.

    Args:
        data_dict: Dictionary of processed DataFrames by location
        combined_data: Combined data for all locations
    """
    if combined_data is not None:
        data_dict["Combined"] = combined_data.copy()


def get_locations_with_combined(base_locations: List[str],
                                data_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Get a list of locations including 'combined' if it exists in the data dictionary.

    Args:
        base_locations: Base list of locations
        data_dict: Dictionary of processed DataFrames by location

    Returns:
        List of locations including 'combined' if available
    """
    locations = base_locations.copy()
    if "Combined" in data_dict:
        locations.append("Combined")
    return locations


def create_tables_for_subject_type(
    data_dict: Dict[str, pd.DataFrame],
    subject_type: str,
    table_dir: Path,
    base_locations: List[str],
    is_dr_site: bool = False
) -> None:
    """
    Create results tables for a specific subject type.

    Args:
        data_dict: Dictionary of processed DataFrames by location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        table_dir: Path to table directory
        base_locations: Base list of locations
        is_dr_site: Whether this is for DR sites
    """
    locations = get_locations_with_combined(base_locations, data_dict)
    create_results_table(data_dict, subject_type, table_dir, locations, is_dr_site)


def create_jasp_files(
    data_dict: Dict[str, pd.DataFrame],
    subject_type: str,
    jasp_dir: Path,
    location: str
) -> None:
    """
    Create JASP format files for a specific location and subject type.

    Args:
        data_dict: Dictionary of processed DataFrames by location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        jasp_dir: Path to JASP directory
        location: Location identifier
    """
    if location in data_dict:
        jasp_df = convert_to_jasp_format(data_dict[location], location, subject_type)
        output_path = jasp_dir / f"{location}_{subject_type}_jasp.csv"
        jasp_df.to_csv(output_path, index=False)


def create_dr_tables(data_dict: Dict[str, pd.DataFrame],
                     subject_type: str, table_dir: Path) -> None:
    """
    Create results tables for DR sites (DR1, DR2) if they exist in the data dictionary.

    Args:
        data_dict: Dictionary of processed DataFrames by location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        table_dir: Path to table directory
    """
    dr_locations = [loc for loc in ["DR1", "DR2"] if loc in data_dict]

    if dr_locations:
        create_tables_for_subject_type(data_dict, subject_type, table_dir,
                                       dr_locations, is_dr_site=True)

def analyze_rt_differences(metrics: Dict[str, List[Dict]]) -> None:
    """
    Analyze and print statistical comparisons of RTs and SSRTs across locations.

    Args:
        metrics: Dictionary containing metrics for each location
    """
    locations = ["Stanford", "Tel Aviv", "UNC"]

    # Extract RTs and SSRTs for each location
    rt_data = {}
    for location in locations:
        if location in metrics:
            rt_data[location] = {
                'stopfail_rt': [m['p2_stopfail_RT'] for m in metrics[location]],
                'go_rt': [m['p2_go_RT'] for m in metrics[location]],
                'go_stop_shapes_rt': ([m['p2_goRT_stop_shapes']
                                       for m in metrics[location]]),
                'go_go_shapes_rt': [m['p2_goRT_go_shapes'] for m in metrics[location]]
            }
    # Compare RTs within each location
    for location in locations:
        if location in rt_data:

            # Compare stop-failure RT with other RTs
            stopfail_rt = rt_data[location]['stopfail_rt']
            go_rt = rt_data[location]['go_rt']
            go_stop_shapes_rt = rt_data[location]['go_stop_shapes_rt']
            go_go_shapes_rt = rt_data[location]['go_go_shapes_rt']

            # Stop-failure RT vs Go RT (all shapes)
            t_stat, p_val1 = stats.ttest_rel(stopfail_rt, go_rt)
            # Stop-failure RT vs Go RT (stop shapes)
            t_stat, p_val2 = stats.ttest_rel(stopfail_rt, go_stop_shapes_rt)
            # Stop-failure RT vs Go RT (go shapes)
            t_stat, p_val3 = stats.ttest_rel(stopfail_rt, go_go_shapes_rt)

            if p_val1 < 0.001 and p_val2 < 0.001 and p_val3 < 0.001:
                print("All p-values comparing Go RT to" +
                      f" Stop-failure RT are less than 0.001 for {location}")
            else:
                print(f"At least one p-value is greater than 0.001 for {location}")

def analyze_iid_effects_by_site(
    data_dir: Path,
    excluded_subjects: Dict[str, List[Dict]] = None,
    output_dir: Path = None,
    table_dir: Path = None,
    figure_dir: Path = None,
    combined_phase1_data: pd.DataFrame = None,
    combined_all_data: pd.DataFrame = None,
    combined_included_data: pd.DataFrame = None
):
    """
    Analyze IID effects by site, counting subjects with positive and negative effects,
    and create summary tables of results.

    Args:
        data_dir: Path to data directory
        excluded_subjects: Dictionary of excluded subjects by location
        output_dir: Path to output directory
        table_dir: Path to table directory
        figure_dir: Path to figure directory
        combined_phase1_data: Combined phase 1 explicit knowledge data
        combined_all_data: Combined all data
        combined_included_data: Combined included data
    """
    jasp_dir = output_dir / "csvs_for_jasp"
    jasp_dir.mkdir(parents=True, exist_ok=True)

    # Process data for different subject types
    data_included = get_processed_data(data_dir, excluded_subjects, "included_only")
    data_all = get_processed_data(data_dir, excluded_subjects, "all")
    data_phase1 = get_processed_data(data_dir, excluded_subjects, "phase1_explicit")

    # Add combined data to respective dictionaries if provided
    add_combined_data(data_included, combined_included_data)
    add_combined_data(data_all, combined_all_data)
    add_combined_data(data_phase1, combined_phase1_data)

    # Create results tables for main sites (Stanford, Tel Aviv, UNC, combined)
    main_locations = ["Stanford", "Tel Aviv", "UNC"]
    create_tables_for_subject_type(data_included, "included", table_dir, main_locations)
    create_tables_for_subject_type(data_all, "all", table_dir, main_locations)
    create_tables_for_subject_type(data_phase1, "phase1", table_dir, main_locations)

    # Create results tables for DR sites (DR1, DR2)
    create_dr_tables(data_included, "included", table_dir)
    create_dr_tables(data_all, "all", table_dir)

    # Process subject files and analyze RT differences
    all_metrics = {}
    for location in main_locations:
        all_metrics[location], _ = process_subject_files(data_dir,
                                                         location,
                                                         excluded_subjects.get(
                                                             location, []
                                                             ))

    # Analyze and print RT differences
    analyze_rt_differences(all_metrics)

    # Create devaluation figures for included subjects and
    # JASP format files for all sites
    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith("."):
            continue

        location = location_dir.name

        # Create devaluation figures for included subjects
        create_combined_devaluation_figures(data_included,
                                            figure_dir,
                                            "included",
                                            include_dr=True,
                                            main_figure_name="figure1.png",
                                            dr_figure_name="figureS3.png")
        create_combined_devaluation_figures(data_all,
                                            figure_dir,
                                            "all",
                                            include_dr=True,
                                            main_figure_name="figureS1.png",
                                            dr_figure_name="figureS4.png")
        create_combined_devaluation_figures(data_phase1,
                                            figure_dir,
                                            "phase1",
                                            include_dr=False,
                                            main_figure_name="figureS2.png")

        # Create JASP format files for all subject types
        create_jasp_files(data_included, "included", jasp_dir, location)
        create_jasp_files(data_all, "all", jasp_dir, location)
        create_jasp_files(data_phase1, "phase1", jasp_dir, location)

    # Also create JASP format files for the combined data
    create_jasp_files(data_included, "included", jasp_dir, "Combined")
    create_jasp_files(data_all, "all", jasp_dir, "Combined")
    create_jasp_files(data_phase1, "phase1", jasp_dir, "Combined")

def average_bidding_across_sites(data: Dict[str, pd.DataFrame],
                                 output_dir: Path,
                                 main_sites: list = ["Stanford",
                                                     "Tel Aviv",
                                                     "UNC"]) -> pd.DataFrame:
    """
    Compute the average bidding level for each value level
    (L, LM, HM, H) across all main sites.

    Args:
        data: Dictionary of processed DataFrames by location.
        main_sites: List of main site names to include.

    Returns:
        DataFrame with index as VALUE_LEVEL and columns as
        average bidding level (mean and std).
    """
    # Concatenate data from all main sites
    dfs = [data[site] for site in main_sites if site in data]
    combined = pd.concat(dfs, ignore_index=True)

    # Group by VALUE_LEVEL and compute mean and std
    summary = combined.groupby("VALUE_LEVEL")["BIDDING_LEVEL"].agg([
        'mean', 'std']).reset_index()
    #make the avg_bidding_levels directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(f"{output_dir}/avg.csv", index=False)

def average_bidding_for_stanford(data:Dict[str, pd.DataFrame],
                                 output_dir:Path) -> pd.DataFrame:
    """
    Compute the average bidding level for each value level (L, LM, HM, H) and
    stop condition (Stop, No stop) for Stanford.

    Args:
        data: Dictionary of processed DataFrames by location
        output_dir: Path to the output directory.

    Returns:
        DataFrame with VALUE_LEVEL, STOP_CONDITION, mean, std, and count columns.
    """
    if "Stanford" not in data:
        raise ValueError("Stanford data not found in data dictionary.")
    df = data["Stanford"].copy()
    # Ensure correct order for VALUE_LEVEL
    value_level_order = pd.CategoricalDtype(["L", "LM", "HM", "H"], ordered=True)
    df["VALUE_LEVEL"] = df["VALUE_LEVEL"].astype(value_level_order)
    # Group by VALUE_LEVEL and STOP_CONDITION
    summary = (
        df.groupby(["VALUE_LEVEL", "STOP_CONDITION"], observed=True)["BIDDING_LEVEL"]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Save to CSV
    summary.to_csv(output_dir, index=False)
    return summary
