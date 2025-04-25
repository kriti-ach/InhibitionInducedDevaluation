from pathlib import Path

from inhibition_induced_devaluation.utils.utils import (
    analyze_iid_effects_by_site,
    combine_location_data,
    create_figure2_and_s2,
    create_stopping_results_tables,
    get_both_exclusions,
    get_iqr_exclusions,
    get_processed_data,
)


def main():
    """Process behavioral data and identify subjects to exclude."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    figure_dir = data_dir.parent / "figures"
    output_dir = data_dir.parent / "output"
    table_dir = data_dir.parent / "tables"

    # Get behavioral exclusions
    behavioral_exclusions = get_both_exclusions(data_dir)
    # Get IQR-based exclusions
    iqr_exclusions = get_iqr_exclusions(
        data_dir,
        {
            loc: [subj["subject_id"] for subj in excl]
            for loc, excl in behavioral_exclusions.items()
        },
    )
    # Create Figure 2
    create_figure2_and_s2(data_dir, figure_dir)

    # Save exclusion summary
    for location, excluded_subjects in behavioral_exclusions.items():
        with open(output_dir / "exclusion_summaries" /
                  f"{location}_exclusion_summary.txt", "a") as f:
            behavioral_count = len(excluded_subjects)
            iqr_count = len(iqr_exclusions.get(location, []))
            total_count = behavioral_count + iqr_count
            f.write(f"IQR-based exclusions: {iqr_count}\n")
            f.write(f"Total excluded subjects: {total_count}\n")

    # Combine behavioral and IQR exclusions for included_data
    all_exclusions = {
        loc: [{'subject_id': subj['subject_id']} for subj in behavioral_exclusions[loc]]
        + [{'subject_id': subj['subject_id']} for subj in iqr_exclusions.get(loc, [])]
        for loc in behavioral_exclusions.keys()
    }

    # Get processed data for different subject groups
    all_data = get_processed_data(data_dir)
    included_data = get_processed_data(
        data_dir, all_exclusions, subject_filter="included_only"
    )
    phase1_data = get_processed_data(data_dir, subject_filter="phase1_explicit")

    # Create stopping results tables
    create_stopping_results_tables(data_dir, table_dir, all_exclusions)
    # Store all equivalence results

    # --- Combined Analyses ---
    # Combined phase 1 explicit subjects analysis
    combined_phase1_data = combine_location_data(phase1_data)
    # # --- Combined All Subjects Analysis ---
    combined_all_data = combine_location_data(all_data)
    # # --- Combined Included Subjects Analysis -----")
    combined_included_data = combine_location_data(included_data)

    analyze_iid_effects_by_site(data_dir, all_exclusions, output_dir,
                                table_dir, figure_dir, combined_phase1_data,
                                combined_all_data, combined_included_data)
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
