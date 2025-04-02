import pandas as pd

from inhibition_induced_devaluation.utils.utils import compute_equivalence_test


def test_compute_equivalence_test_basic():
    """Test basic equivalence test computation"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'No Stop', 'Stop', 'No Stop'],
        'BIDDING_LEVEL': [4, 3, 5, 4]
    }
    df = pd.DataFrame(data)

    results = compute_equivalence_test(df)
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'N', 'Mean_Difference', 'SD_Difference', 'Equivalence_Margin',
        'TOST_lower_p', 'TOST_upper_p', 'Equivalent'
    ])

def test_compute_equivalence_test_no_difference():
    """Test equivalence test with no difference between conditions"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'No Stop', 'Stop', 'No Stop'],
        'BIDDING_LEVEL': [4, 4, 5, 5]
    }
    df = pd.DataFrame(data)

    results = compute_equivalence_test(df)
    assert results['Mean_Difference'] == 0
    assert results['Equivalent']
