from inhibition_induced_devaluation.utils.utils import (
    get_behavioral_exclusions,
    get_data_dir,
    get_iqr_exclusions,
)


def main():
    """Process behavioral data and identify subjects to exclude."""
    data_dir = get_data_dir()

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

        # Group by reason
        reasons = {}
        for subject in excluded_subjects:
            reason = subject['detailed_reason']
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(subject['subject_id'])

        # Add IQR exclusions if any exist for this location
        if location in iqr_exclusions:
            iqr_reason = 'IID effect outside 1.5*IQR range'
            reasons[iqr_reason] = [
                subject['subject_id'] for subject in iqr_exclusions[location]
            ]

        # Print detailed breakdown
        for reason, subjects in reasons.items():
            print(f'\n{reason}:')
            print(f'Subjects: {", ".join(sorted(subjects))}')
            print(f'Count: {len(subjects)}')


if __name__ == '__main__':
    main()
