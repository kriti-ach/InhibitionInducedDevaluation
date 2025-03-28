from inhibition_induced_devaluation.utils.utils import (
    get_behavioral_exclusions,
    get_data_dir,
    get_iqr_exclusions,
    get_processed_data,
    create_devaluation_figure,
    perform_rm_anova,
    combine_location_data,
    perform_equivalence_testing,
)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd


def main():
    """Process behavioral data and identify subjects to exclude."""
    data_dir = get_data_dir()
    figure_dir = data_dir.parent / 'figures'
    output_dir = data_dir.parent / 'output'
    table_dir = data_dir.parent / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    
    # Get behavioral exclusions first
    behavioral_exclusions = get_behavioral_exclusions(data_dir)
    
    # Get IQR exclusions only for non-excluded subjects
    iqr_exclusions = get_iqr_exclusions(
        data_dir,
        {
            loc: [subj['subject_id'] for subj in excl]
            for loc, excl in behavioral_exclusions.items()
        },
    )
    
    # Print exclusion summary
    for location, excluded_subjects in behavioral_exclusions.items():
        print(f'\nExclusions for {location}:')
        behavioral_count = len(excluded_subjects)
        iqr_count = len(iqr_exclusions.get(location, []))
        total_count = behavioral_count + iqr_count
        
        print(f'Total excluded subjects: {total_count}')
        print(f'Behavioral exclusions: {behavioral_count}')
        print(f'IQR-based exclusions: {iqr_count}')
    
    # Combine behavioral and IQR exclusions for included_data
    all_exclusions = {
        loc: [{'subject_id': subj['subject_id']} for subj in behavioral_exclusions[loc]] + 
             [{'subject_id': subj['subject_id']} for subj in iqr_exclusions.get(loc, [])]
        for loc in behavioral_exclusions.keys()
    }
    
    # Get processed data for different subject groups
    all_data = get_processed_data(data_dir)
    included_data = get_processed_data(data_dir, all_exclusions, subject_filter='included_only')
    phase1_data = get_processed_data(data_dir, subject_filter='phase1_explicit')
    
    # Store all equivalence results
    all_equiv_results = []
    
    # Process each dataset and create summary tables
    for location in behavioral_exclusions.keys():
        # Create figures and perform analyses
        create_devaluation_figure(all_data[location], location, 'all', figure_dir)
        perform_rm_anova(all_data[location], location, 'all', output_dir)
        all_equiv_results.append(perform_equivalence_testing(all_data[location], location, 'all'))
        
        create_devaluation_figure(included_data[location], location, 'included', figure_dir)
        perform_rm_anova(included_data[location], location, 'included', output_dir)
        all_equiv_results.append(perform_equivalence_testing(included_data[location], location, 'included'))
        
        if location != 'DR2':
            create_devaluation_figure(phase1_data[location], location, 'phase1', figure_dir)
            perform_rm_anova(phase1_data[location], location, 'phase1', output_dir)
            all_equiv_results.append(perform_equivalence_testing(phase1_data[location], location, 'phase1'))
    
    # Combined phase 1 explicit subjects analysis
    combined_phase1_data = combine_location_data(phase1_data)
    create_devaluation_figure(combined_phase1_data, 'combined', 'phase1', figure_dir)
    perform_rm_anova(combined_phase1_data, 'combined', 'phase1', output_dir)
    all_equiv_results.append(perform_equivalence_testing(combined_phase1_data, 'combined', 'phase1'))
    
    # Save equivalence results summary
    combined_equiv_results = pd.concat(all_equiv_results, ignore_index=True)
    equiv_dir = output_dir / 'equivalence_tests'
    equiv_dir.mkdir(parents=True, exist_ok=True)
    combined_equiv_results.to_csv(equiv_dir / 'equivalence_tests_summary.csv', index=False)


if __name__ == "__main__":
    main()