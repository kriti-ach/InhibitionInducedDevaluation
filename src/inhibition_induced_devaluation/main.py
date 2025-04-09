import pandas as pd
from pathlib import Path

from inhibition_induced_devaluation.utils.utils import (
    combine_location_data,
    create_figure2,
    create_stopping_results_table,
    get_both_exclusions,
    get_iqr_exclusions,
    get_processed_data,
    analyze_iid_effects_by_site,
    run_standard_analyses,
)


def main():
    """Process behavioral data and identify subjects to exclude."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    figure_dir = data_dir.parent / "figures"
    output_dir = data_dir.parent / "output"
    table_dir = data_dir.parent / "tables"
    jasp_dir = output_dir / "csvs_for_jasp"

    # Create necessary directories
    for directory in [table_dir, jasp_dir, figure_dir, output_dir, output_dir / "anovas", output_dir / "equivalence_tests", output_dir / "iid_effect_counts"]:
        directory.mkdir(parents=True, exist_ok=True)
        # Get IQR exclusions only for non-excluded subjects

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
    create_figure2(data_dir, figure_dir)

    # Print exclusion summary
    for location, excluded_subjects in behavioral_exclusions.items():
        with open(output_dir / "exclusion_summaries" / f"{location}_exclusion_summary.txt", "a") as f:
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

    # Create stopping results tables for each location
    for location in behavioral_exclusions.keys():
        # Get list of all excluded subjects for this location
        excluded_subject_ids = [
            subj["subject_id"] for subj in behavioral_exclusions[location]
        ] + [subj["subject_id"] for subj in iqr_exclusions.get(location, [])]

        # Create and save stopping results table
        stopping_table = create_stopping_results_table(
            data_dir, location, excluded_subject_ids
        )
        stopping_table.to_csv(
            table_dir / f"{location}_stopping_results.csv", index=False
        )

    # Store all equivalence results
    all_equiv_results = []

    # Process each dataset and create outputs
    for location in behavioral_exclusions.keys():
        print(f"\n--- Processing Location: {location} ---")
        # --- All Subjects --- 
        equiv_res = run_standard_analyses(
            all_data[location], location, "all", figure_dir, output_dir, jasp_dir
        )
        all_equiv_results.append(equiv_res)

        # --- Included Subjects --- 
        equiv_res = run_standard_analyses(
            included_data[location], location, "included", figure_dir, output_dir, jasp_dir
        )
        all_equiv_results.append(equiv_res)

        # --- Phase 1 Explicit (if applicable) ---
        if location not in ["DR1", "DR2"]:
            equiv_res = run_standard_analyses(
                phase1_data[location], location, "phase1", figure_dir, output_dir, jasp_dir
            )
            all_equiv_results.append(equiv_res)


    # --- Combined Analyses ---
    # Combined phase 1 explicit subjects analysis
    combined_phase1_data = combine_location_data(phase1_data)
    equiv_res = run_standard_analyses(combined_phase1_data, "combined", "phase1", figure_dir, output_dir, jasp_dir)
    all_equiv_results.append(equiv_res)

    # --- Combined All Subjects Analysis ---
    combined_all_data = combine_location_data(all_data)
    equiv_res = run_standard_analyses(combined_all_data, "combined", "all", figure_dir, output_dir, jasp_dir)
    all_equiv_results.append(equiv_res)

    # --- Combined Included Subjects Analysis -----")
    combined_included_data = combine_location_data(included_data)
    equiv_res = run_standard_analyses(combined_included_data, "combined", "included", figure_dir, output_dir, jasp_dir)
    all_equiv_results.append(equiv_res)

    # Save equivalence results summary
    combined_equiv_results = pd.concat(all_equiv_results, ignore_index=True)
    equiv_dir = output_dir / "equivalence_tests" # Directory already created
    combined_equiv_results.to_csv(
        equiv_dir / "equivalence_tests_summary.csv", index=False
    )
    
    analyze_iid_effects_by_site(data_dir, all_exclusions, output_dir)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
