from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.power import tt_solve_power, TTestPower
from scipy import stats

# Define explicit knowledge subjects for each location
EXPLICIT_KNOWLEDGE_SUBJECTS = {
    'DR2': {
        'S91',
        'S97',
        'S100',
        'S102',
        'S106',
        'S108',
        'S112',
        'S32',
        'S49',
        'S87',
        'S90',
        'S92',
        'S95',
        'S96',
        'S98',
        'S103',
        'S110',
    },
    'Stanford': {
        'S768',
        'S770',
        'S772',
        'S774',
        'S776',
        'S778',
        'S779',
        'S780',
        'S782',
        'S783',
        'S786',
        'S791',
        'S792',
        'S795',
        'S796',
        'S798',
        'S801',
        'S807',
        'S808',
        'S811',
        'S814',
        'S815',
        'S818',
        'S819',
        'S822',
        'S825',
        'S826',
        'S827',
        'S833',
        'S836',
        'S846',
        'S849',
        'S854',
        'S858',
        'S868',
        'S875',
        'S876',
        'S882',
        'S887',
        'S889',
        'S891',
        'S895',
        'S896',
        'S901',
        'S906',
        'S908',
        'S912',
        'S834',
    },
    'Tel Aviv': {
        'S142',
        'S148',
        'S150',
        'S152',
        'S154',
        'S156',
        'S158',
        'S162',
        'S163',
        'S165',
        'S166',
        'S167',
        'S168',
        'S169',
        'S173',
        'S176',
        'S180',
        'S188',
        'S192',
        'S193',
        'S200',
        'S203',
        'S205',
        'S208',
        'S215',
        'S216',
        'S223',
        'S224',
        'S225',
        'S230',
        'S231',
        'S235',
        'S242',
        'S247',
        'S248',
        'S250',
        'S256',
        'S262',
        'S271',
        'S277',
        'S279',
        'S280',
        'S282',
        'S284',
        'S286',
        'S292',
        'S296',
        'S298',
        'S300',
        'S301',
        'S302',
        'S304',
        'S309',
        'S311',
        'S313',
    },
    'UNC': {
        'S4001',
        'S4002',
        'S4004',
        'S4005',
        'S4006',
        'S4008',
        'S4012',
        'S4014',
        'S4019',
        'S4022',
        'S4023',
        'S4027',
        'S4028',
        'S4029',
        'S4040',
        'S4043',
        'S4048',
        'S4057',
        'S4058',
        'S4063',
        'S4065',
        'S4069',
        'S4070',
        'S4073',
        'S4074',
        'S4075',
        'S4081',
        'S4085',
        'S4086',
        'S4088',
        'S4095',
        'S4099',
        'S4103',
        'S4108',
        'S4109',
        'S4115',
        'S4116',
        'S4122',
        'S4129',
        'S4131',
        'S4133',
        'S4135',
        'S4138',
        'S4142',
        'S4146',
        'S4149',
        'S4156',
        'S4161',
        'S4163',
        'S4165',
        'S4167',
        'S4168',
        'S4173',
        'S4174',
        'S4175',
        'S4181',
        'S4183',
        'S4185',
        'S4192',
        'S4204',
        'S4211',
        'S4218',
        'S4225',
    },
}
MAX_RT = 1000  # In milliseconds, the maximum reaction time

PHASE1_EXPLICIT_KNOWLEDGE = {
    'Stanford': [
        'S766', 'S790', 'S791', 'S838', 'S841', 'S873',
        'S902', 'S903', 'S904', 'S907', 'S914'
    ],
    'UNC': [
        'S4024', 'S4051', 'S4062', 'S4064', 'S4067', 'S4068',
        'S4077', 'S4079', 'S4090', 'S4096', 'S4123', 'S4124',
        'S4136', 'S4140', 'S4160', 'S4186', 'S4191', 'S4195',
        'S4196', 'S4210', 'S4213', 'S4215', 'S4217', 'S4219',
        'S4221', 'S4222'
    ],
    'Tel Aviv': [
        'S145', 'S157', 'S170', 'S185', 'S187', 'S198',
        'S219', 'S221', 'S236', 'S247', 'S260', 'S261',
        'S274', 'S291', 'S303', 'S305', 'S308'
    ]
}

def get_project_root() -> Path:
    """Return the project root directory as a Path object."""
    return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    """Return the data directory as a Path object."""
    return get_project_root() / 'data'


def get_data_locations() -> List[str]:
    """
    Get all data collection location names from the data directory.

    Returns:
        List[str]: List of directory names representing data collection locations
    """
    data_dir = get_data_dir()
    # Get only directories and filter out hidden directories (starting with .)
    return [
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')
    ]


def load_location_data(location: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a specific data collection location.

    Args:
        location (str): Name of the data collection location (e.g., 'DR2', 'UNC', etc.)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with filenames as keys and pandas DataFrames as values
    """
    data_dir = get_data_dir() / location
    if not data_dir.exists():
        print(f"Warning: Location directory '{location}' does not exist")
        return {}

    csv_files = {}
    for csv_path in data_dir.rglob('*.csv'):
        try:
            df = pd.read_csv(csv_path)
            # Use relative path from the location directory as key
            relative_path = csv_path.relative_to(data_dir)
            csv_files[str(relative_path)] = df
        except Exception as e:
            print(f'Error loading {csv_path}: {str(e)}')

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


# Keep the original function for backward compatibility
def load_csv_files(subdirectory: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the data directory or a specific subdirectory.

    Args:
        subdirectory (str, optional): Name of subdirectory within data folder to search.
                                    If None, searches the main data directory.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with filenames as keys and pandas DataFrames as values
    """
    data_dir = get_data_dir()
    if subdirectory:
        data_dir = data_dir / subdirectory

    csv_files = {}
    for csv_path in data_dir.rglob('*.csv'):
        try:
            df = pd.read_csv(csv_path)
            # Use relative path from data directory as key
            relative_path = csv_path.relative_to(get_data_dir())
            csv_files[str(relative_path)] = df
        except Exception as e:
            print(f'Error loading {csv_path}: {str(e)}')

    return csv_files


def process_stop_signal_data(
    subject_id, df: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """Process stop signal data to calculate key metrics."""
    # Filter for part 2 only
    df_p2 = df[df['which_part'] == 'part_2'].copy()

    # Get trial types
    stop_failure_trials = df_p2.loc[df_p2['accuracy'] == 3]
    no_stop_signal_trials_stop_shapes = df_p2.loc[
        (df_p2['paired_with_stopping'] == 1) & (df_p2['stop_signal_trial_type'] == 'go')
    ].copy()

    # Calculate mean RTs
    meanRT_stop_fail = stop_failure_trials['reaction_time'].mean()
    meanRT_no_stop_trials_stop_shapes = no_stop_signal_trials_stop_shapes[
        'reaction_time'
    ].mean()

    # Process SSRT calculation
    trials_with_SS = df_p2.loc[df_p2['stop_signal_trial_type'] == 'stop']
    trials_with_SS_left = trials_with_SS.loc[trials_with_SS['quadrant'] == 5]
    trials_with_SS_right = trials_with_SS.loc[trials_with_SS['quadrant'] == 6]

    # Calculate probability of responding
    successful_stops = trials_with_SS.groupby('response').count().iloc[0].accuracy
    pRespond_given_SS = (len(trials_with_SS) - successful_stops) / len(trials_with_SS)
    # Process no-stop trials
    rank = round(pRespond_given_SS * len(no_stop_signal_trials_stop_shapes))
    rank_left_trials = no_stop_signal_trials_stop_shapes.loc[
        no_stop_signal_trials_stop_shapes['quadrant'] == 5
    ].copy()
    rank_right_trials = no_stop_signal_trials_stop_shapes.loc[
        no_stop_signal_trials_stop_shapes['quadrant'] == 6
    ].copy()

    # We are replacing missed no-stop trials with the maximum response time, 1 second
    no_stop_signal_trials_stop_shapes['reaction_time_replaced'] = np.where(
        no_stop_signal_trials_stop_shapes['reaction_time'] == 0,
        MAX_RT,
        no_stop_signal_trials_stop_shapes['reaction_time'],
    )

    # Calculate SSRT components
    if len(no_stop_signal_trials_stop_shapes['reaction_time_replaced']) <= rank:
        return (
            meanRT_stop_fail,
            meanRT_no_stop_trials_stop_shapes,
            float('nan'),
            float('nan'),
        )

    Nth_RT = (
        no_stop_signal_trials_stop_shapes.sort_values(by=['reaction_time_replaced'])
        .iloc[int(rank)]
        .reaction_time_replaced
    )

    avg_SSD = (
        rank_left_trials['left_SSD'].mean() + rank_right_trials['right_SSD'].mean()
    ) / 2
    SSRT = Nth_RT - avg_SSD

    return meanRT_stop_fail, meanRT_no_stop_trials_stop_shapes, SSRT, avg_SSD


def check_exclusion_criteria(
    meanRT_stop_fail: float,
    meanRT_no_stop_trials_stop_shapes: float,
    SSRT: float,
    min_SSRT_cutoff: float = 100,
) -> Tuple[List[int], str]:
    """Check if a subject meets exclusion criteria."""
    subject_vector = []
    # Check stop-failure RT criterion
    if meanRT_stop_fail >= meanRT_no_stop_trials_stop_shapes:
        subject_vector.append(0)
    else:
        subject_vector.append(1)

    # Check SSRT criterion
    if np.isnan(SSRT) or SSRT < min_SSRT_cutoff:
        subject_vector.append(0)
    else:
        subject_vector.append(1)

    # Determine exclusion reason
    if subject_vector == [1, 1]:
        reason = 'include - subject passed all criteria'
    elif subject_vector == [1, 0]:
        reason = 'exclude - SSRT is lower than the minimum SSRT'
    elif subject_vector == [0, 1]:
        reason = 'exclude - stop fail RT is >= no-stop RT'
    else:
        reason = 'exclude - failed both criteria'

    return subject_vector, reason


def add_explicit_knowledge_exclusions(
    location: str, behavioral_exclusions: List[Dict]
) -> List[Dict]:
    """Add explicit knowledge exclusions to the behavioral exclusions list."""
    if location not in EXPLICIT_KNOWLEDGE_SUBJECTS:
        return behavioral_exclusions

    # Convert behavioral exclusions to a dict for easy lookup
    excluded_subjects = {exc['subject_id']: exc for exc in behavioral_exclusions}

    for subject in EXPLICIT_KNOWLEDGE_SUBJECTS[location]:
        if subject in excluded_subjects:
            # Update reason if subject was already excluded for behavioral reasons
            if excluded_subjects[subject]['reason'] == 'Behavior':
                excluded_subjects[subject]['reason'] = 'Behavior and Explicit Knowledge'
                excluded_subjects[subject]['detailed_reason'] = (
                    f'Failed behavioral criteria ({excluded_subjects[subject]["detailed_reason"]}) and reported explicit knowledge'
                )
        else:
            # Add new exclusion for explicit knowledge
            exclusion = {
                'subject_id': subject,
                'reason': 'Explicit Knowledge',
                'detailed_reason': 'Explicit Knowledge',
            }
            behavioral_exclusions.append(exclusion)

    # Special case for Stanford S834
    if location == 'Stanford' and 'S834' in EXPLICIT_KNOWLEDGE_SUBJECTS[location]:
        if 'S834' not in excluded_subjects:
            behavioral_exclusions.append(
                {
                    'subject_id': 'S834',
                    'reason': 'Behavior',
                    'detailed_reason': 'SSD reached 0 and stayed there',
                }
            )

    return sorted(behavioral_exclusions, key=lambda x: x['subject_id'])


def process_csv_file(file_path: Path, dataset_collection_place: str) -> Dict:
    """Process a single CSV file and return exclusion data."""
    try:
        df = pd.read_csv(file_path)
        subject_id = file_path.stem

        # Special case for Stanford S819
        if dataset_collection_place.lower() == 'stanford' and 'S819' in file_path.name:
            return {
                'subject_id': subject_id,
                'reason': 'Behavior',
                'detailed_reason': 'Did not stop',
            }

        # Process behavioral data
        meanRT_stop_fail, meanRT_no_stop_trials_stop_shapes, SSRT, avg_SSD = (
            process_stop_signal_data(subject_id, df)
        )
        # Check exclusion criteria
        subject_vector, reason = check_exclusion_criteria(
            meanRT_stop_fail, meanRT_no_stop_trials_stop_shapes, SSRT
        )

        if subject_vector != [1, 1]:  # If subject should be excluded
            return {
                'subject_id': subject_id,
                'reason': 'Behavior',
                'detailed_reason': reason,
                'metrics': {
                    'meanRT_stop_fail': meanRT_stop_fail,
                    'meanRT_no_stop_trials_stop_shapes': meanRT_no_stop_trials_stop_shapes,
                    'SSRT': SSRT,
                    'avg_SSD': avg_SSD,
                },
            }
        return {}

    except Exception as e:
        print(f'Error processing {file_path}: {str(e)}')
        return {}


def get_behavioral_exclusions(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Process all CSV files in each location directory and return exclusion data.

    Returns:
        Dict[str, List[Dict]]: Dictionary with locations as keys and lists of excluded subjects as values
    """
    exclusions = {}

    for location_dir in data_dir.iterdir():
        if not location_dir.is_dir() or location_dir.name.startswith('.'):
            continue

        location_exclusions = []
        csv_files = sorted(list(location_dir.glob('*.csv')))

        if not csv_files:
            print(f'No CSV files found in {location_dir}')
            continue

        print(f'Processing {len(csv_files)} files in {location_dir.name}')

        for csv_file in csv_files:
            exclusion_data = process_csv_file(csv_file, location_dir.name)
            if exclusion_data:
                location_exclusions.append(exclusion_data)

        if location_exclusions:
            # Add explicit knowledge exclusions
            location_exclusions = add_explicit_knowledge_exclusions(
                location_dir.name, location_exclusions
            )
            exclusions[location_dir.name] = location_exclusions
    return exclusions


def process_phase3_data(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Process phase 3 data to calculate IID effects."""
    df_p3 = df[df['which_part'] == 'part_3'].copy()

    stop_shapes = df_p3[df_p3['paired_with_stopping'] == 1]
    go_shapes = df_p3[df_p3['paired_with_stopping'] == 0]

    stop_bid = stop_shapes['chosen_bidding_level'].mean()
    go_bid = go_shapes['chosen_bidding_level'].mean()
    iid_effect = go_bid - stop_bid  # Positive values indicate devaluation

    return iid_effect, stop_bid, go_bid


def calculate_iqr_cutoffs(iid_effects: List[float]) -> Tuple[float, float]:
    """Calculate IQR-based cutoffs for outlier detection."""
    q75, q25 = np.percentile(iid_effects, [75, 25])
    iqr = q75 - q25
    upper_cutoff = q75 + iqr * 1.5
    lower_cutoff = q25 - iqr * 1.5
    return upper_cutoff, lower_cutoff


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
        if not location_dir.is_dir() or location_dir.name.startswith('.'):
            continue

        # Get list of already excluded subjects for this location
        already_excluded = excluded_subjects.get(location_dir.name, [])

        # Process each location's data
        iid_effects = []
        subject_data = {}
        csv_files = sorted(list(location_dir.glob('*.csv')))

        if not csv_files:
            print(f'No CSV files found in {location_dir}')
            continue

        print(
            f'\nProcessing {len(csv_files)} files in {location_dir.name} for IQR exclusions'
        )

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
                print(f'Error processing {csv_file}: {str(e)}')

        if not iid_effects:
            print(
                f'No valid IID effects found for non-excluded subjects in {location_dir.name}'
            )
            continue

        # Calculate cutoffs and identify outliers
        upper_cutoff, lower_cutoff = calculate_iqr_cutoffs(iid_effects)
        location_exclusions = []

        for subject_id, (iid_effect, stop_bid, go_bid) in subject_data.items():
            if iid_effect > upper_cutoff or iid_effect < lower_cutoff:
                exclusion = {
                    'subject_id': subject_id,
                    'reason': 'IID Effect',
                    'detailed_reason': 'IID effect outside 1.5*IQR range',
                    'metrics': {
                        'iid_effect': iid_effect,
                        'stop_bid': stop_bid,
                        'go_bid': go_bid,
                        'upper_cutoff': upper_cutoff,
                        'lower_cutoff': lower_cutoff,
                    },
                }
                location_exclusions.append(exclusion)

        if location_exclusions:
            exclusions[location_dir.name] = location_exclusions

    return exclusions


def process_subject_data(file_path: Path) -> pd.DataFrame:
    """Process individual subject data file."""
    df = pd.read_csv(file_path)
    subject_id = file_path.stem  # Gets filename without extension
    
    df['BIDDING_LEVEL'] = df['chosen_bidding_level']
    value_level_mapping = {1: 'L', 2: 'LM', 3: 'HM', 4: 'H'}
    df['VALUE_LEVEL'] = df['stepwise_reward_magnitude'].map(value_level_mapping)
    df['SUBJECT'] = subject_id
    
    df_agg = df.groupby(['SUBJECT', 'VALUE_LEVEL', 'paired_with_stopping'])['BIDDING_LEVEL'].mean().reset_index()
    df_agg['STOP_CONDITION'] = df_agg['paired_with_stopping'].map({0: 'No Stop', 1: 'Stop'})
    
    return df_agg[['SUBJECT', 'VALUE_LEVEL', 'STOP_CONDITION', 'BIDDING_LEVEL']]


def get_processed_data(data_dir: Path, 
                      excluded_subjects: Dict[str, List[str]] = None,
                      subject_filter: str = 'all') -> Dict[str, pd.DataFrame]:
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
        if not location_dir.is_dir() or location_dir.name.startswith('.'):
            continue
            
        location = location_dir.name
        csv_files = sorted(list(location_dir.glob('*.csv')))
        
        if not csv_files:
            print(f"No CSV files found in {location}")
            continue
            
        # Determine which subjects to process based on filter
        excluded = {subj['subject_id'] for subj in excluded_subjects[location]} if location in excluded_subjects else set()
        phase1_explicit = set(PHASE1_EXPLICIT_KNOWLEDGE.get(location, []))
        
        dfs = []
        for file_path in csv_files:
            subject_id = file_path.stem
            
            # Apply filters
            if subject_filter == 'included_only' and subject_id in excluded:
                continue
            elif subject_filter == 'phase1_explicit' and subject_id not in phase1_explicit:
                continue
                
            try:
                df = process_subject_data(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if dfs:
            processed_data[location] = pd.concat(dfs, ignore_index=True)
        else:
            print(f"No valid data processed for {location}")
    
    return processed_data


def create_devaluation_figure(df: pd.DataFrame, location: str, subject_type: str, figure_dir: Path) -> None:
    """
    Create devaluation figure from processed data.
    
    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        figure_dir: Path to figures directory
    """
    # Define the desired order for 'VALUE_LEVEL'
    value_level_order = pd.CategoricalDtype(['L', 'LM', 'HM', 'H'], ordered=True)
    
    # Convert 'VALUE_LEVEL' to categorical type with defined order
    df['VALUE_LEVEL'] = df['VALUE_LEVEL'].astype(value_level_order)
    
    # Calculate statistics
    group_stats = df.groupby(['VALUE_LEVEL', 'STOP_CONDITION'], observed=True)['BIDDING_LEVEL'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Calculate SEM
    group_stats['SEM'] = group_stats['std'] / np.sqrt(group_stats['count'])
    
    # Pivot data
    avg_pivot = group_stats.pivot(index='VALUE_LEVEL', columns='STOP_CONDITION', values='mean')
    sem_pivot = group_stats.pivot(index='VALUE_LEVEL', columns='STOP_CONDITION', values='SEM')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    avg_pivot.plot(kind='bar', yerr=sem_pivot, capsize=4)
    
    plt.title(f'{location} - {subject_type} subjects')
    plt.xlabel('Value Level')
    plt.ylabel('Average Bidding Level')
    plt.xticks(rotation=0)
    plt.yticks(np.arange(1, 7, 1))
    plt.legend(title='Stop Condition')
    
    # Create directory if it doesn't exist
    location_dir = figure_dir / location
    location_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(location_dir / f'{location}_{subject_type}_devaluation_figure.png')
    plt.close()


def perform_rm_anova(df: pd.DataFrame, location: str, subject_type: str, output_dir: Path) -> None:
    """
    Perform repeated measures ANOVA on the data and save results.
    
    Args:
        df: Processed DataFrame with bidding data
        location: Collection location
        subject_type: Type of subjects ('all', 'included', or 'phase1')
        output_dir: Path to output directory
    """
    # Ensure categorical variables
    df['STOP_CONDITION'] = df['STOP_CONDITION'].astype('category')
    df['VALUE_LEVEL'] = df['VALUE_LEVEL'].astype('category')
    
    # Perform repeated measures ANOVA
    anova_results = AnovaRM(
        data=df,
        depvar='BIDDING_LEVEL',
        subject='SUBJECT',
        within=['STOP_CONDITION', 'VALUE_LEVEL']
    ).fit()
    
    # Create anovas directory if it doesn't exist
    anova_dir = output_dir / 'anovas'
    anova_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to file
    output_file = anova_dir / f'{location}_{subject_type}_rm_anova_results.txt'
    with open(output_file, 'w') as f:
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
        df_copy['LOCATION'] = location
        dfs.append(df_copy)
    
    return pd.concat(dfs, ignore_index=True)


def perform_equivalence_testing(df: pd.DataFrame, location: str, subject_type: str, equivalence_margin: float = 0.5) -> pd.DataFrame:
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
    df['STOP_CONDITION'] = df['STOP_CONDITION'].astype('category')
    
    # Aggregate BIDDING_LEVEL by SUBJECT and STOP_CONDITION
    aggregated_df = df.groupby(['SUBJECT', 'STOP_CONDITION'], observed=True)['BIDDING_LEVEL'].mean().reset_index()
    
    # Pivot to align paired observations
    paired_df = aggregated_df.pivot(index='SUBJECT', 
                                  columns='STOP_CONDITION', 
                                  values='BIDDING_LEVEL').dropna()
    
    stop_group = paired_df['Stop']
    no_stop_group = paired_df['No Stop']
    
    # Calculate statistics
    diff = no_stop_group - stop_group
    n = len(diff)
    
    # Lower bound test (null: diff <= -margin)
    t_lower = (np.mean(diff) + equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_lower = 1 - stats.t.cdf(t_lower, df=n-1)
    
    # Upper bound test (null: diff >= margin)
    t_upper = (np.mean(diff) - equivalence_margin) / (np.std(diff, ddof=1) / np.sqrt(n))
    p_upper = stats.t.cdf(t_upper, df=n-1)
    
    # Create results DataFrame
    return pd.DataFrame([{
        'Location': location,
        'Subject_Type': subject_type,
        'N': n,
        'Mean_Difference': np.mean(diff),
        'SD_Difference': np.std(diff, ddof=1),
        'Equivalence_Margin': equivalence_margin,
        'TOST_lower_p': p_lower,
        'TOST_upper_p': p_upper,
        'Equivalent': p_lower < 0.05 and p_upper < 0.05
    }])
