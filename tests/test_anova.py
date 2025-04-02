import pandas as pd
import pytest

from inhibition_induced_devaluation.utils.utils import compute_rm_anova


def test_compute_rm_anova_basic():
    """Test basic ANOVA computation with duplicate observations"""
    # Create sample data with duplicates
    data = {
        'SUBJECT': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'Stop', 'No Stop', 'No Stop'] * 2,
        'VALUE_LEVEL': ['H', 'L', 'H', 'L'] * 2,
        'BIDDING_LEVEL': [4, 2, 3, 1, 5, 3, 4, 2]
    }
    df = pd.DataFrame(data)

    # Add duplicate rows with slightly different values
    duplicate_data = data.copy()
    duplicate_data['BIDDING_LEVEL'] = [x + 0.5 for x in data['BIDDING_LEVEL']]
    df = pd.concat([df, pd.DataFrame(duplicate_data)], ignore_index=True)

    results = compute_rm_anova(df)
    assert results is not None
    assert 'STOP_CONDITION' in results.anova_table.index
    assert 'VALUE_LEVEL' in results.anova_table.index

def test_compute_rm_anova_no_duplicates():
    """Test ANOVA computation with no duplicate observations"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'Stop', 'No Stop', 'No Stop'] * 2,
        'VALUE_LEVEL': ['H', 'L', 'H', 'L'] * 2,
        'BIDDING_LEVEL': [4, 2, 3, 1, 5, 3, 4, 2]
    }
    df = pd.DataFrame(data)

    results = compute_rm_anova(df)
    assert results is not None
    assert 'STOP_CONDITION' in results.anova_table.index
    assert 'VALUE_LEVEL' in results.anova_table.index

def test_compute_rm_anova_missing_data():
    """Test ANOVA computation with missing data"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'Stop', 'No Stop'] * 2,
        'VALUE_LEVEL': ['H', 'L', 'H'] * 2,
        'BIDDING_LEVEL': [4, 2, 3, 5, 3, 4]
    }
    df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        compute_rm_anova(df)
