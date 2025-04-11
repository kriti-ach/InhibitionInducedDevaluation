import pandas as pd
import pytest
import numpy as np
from inhibition_induced_devaluation.utils.utils import perform_equivalence_testing

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
        'STOP_CONDITION': ['Stop', 'No Stop'] * 3,
        'BIDDING_LEVEL': [3, 3, 4, 4, 5, 5],  # No difference between conditions
        'VALUE_LEVEL': ['H', 'H', 'H', 'H', 'H', 'H']
    }
    return pd.DataFrame(data)

def test_perform_equivalence_testing_basic(sample_data):
    """Test basic equivalence testing functionality"""
    results = perform_equivalence_testing(sample_data, "TEST", "all")
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1  # Should return one row of results
    
    # Check required columns exist
    required_columns = [
        'Location', 'Subject_Type', 'N', 'Mean_Difference',
        'SD_Difference', 'Equivalence_Margin',
        'TOST_lower_t', 'TOST_upper_t',
        'TOST_lower_p', 'TOST_upper_p', 'Equivalent'
    ]
    assert all(col in results.columns for col in required_columns)

def test_perform_equivalence_testing_no_difference(sample_data):
    """Test equivalence testing with no difference between conditions"""
    results = perform_equivalence_testing(sample_data, "TEST", "all")
    
    assert results['Mean_Difference'].iloc[0] == 0
    assert results['Equivalent'].iloc[0] == True

def test_perform_equivalence_testing_with_difference():
    """Test equivalence testing with clear difference between conditions"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'No Stop'] * 2,
        'BIDDING_LEVEL': [1, 5, 2, 6],  # Large difference between conditions
        'VALUE_LEVEL': ['H', 'H', 'H', 'H']
    }
    df = pd.DataFrame(data)
    
    results = perform_equivalence_testing(df, "TEST", "all")
    assert results['Equivalent'].iloc[0] == False
    assert abs(results['Mean_Difference'].iloc[0]) > results['Equivalence_Margin'].iloc[0]

def test_perform_equivalence_testing_custom_margin():
    """Test equivalence testing with custom equivalence margin"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'No Stop'] * 2,
        'BIDDING_LEVEL': [3, 4, 3, 4],  # Small difference between conditions
        'VALUE_LEVEL': ['H', 'H', 'H', 'H']
    }
    df = pd.DataFrame(data)
    
    # Test with different equivalence margins
    results_small = perform_equivalence_testing(df, "TEST", "all", equivalence_margin=0.5)
    results_large = perform_equivalence_testing(df, "TEST", "all", equivalence_margin=2.0)
    
    # Smaller margin should be less likely to show equivalence
    assert results_small['Equivalence_Margin'].iloc[0] == 0.5
    assert results_large['Equivalence_Margin'].iloc[0] == 2.0
    assert results_large['Equivalent'].iloc[0] == True  # Should be equivalent with large margin

def test_perform_equivalence_testing_invalid_data():
    """Test equivalence testing with invalid data"""
    # Test with empty DataFrame
    empty_df = pd.DataFrame({
        'SUBJECT': [],
        'STOP_CONDITION': [],
        'BIDDING_LEVEL': [],
        'VALUE_LEVEL': []
    })
    
    with pytest.raises(Exception):  # Should raise an error due to insufficient data
        perform_equivalence_testing(empty_df, "TEST", "all") 