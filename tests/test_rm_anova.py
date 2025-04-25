
import pandas as pd
import pytest

from inhibition_induced_devaluation.utils.utils import perform_rm_anova


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'SUBJECT': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2'],
        'STOP_CONDITION': ['Stop', 'Stop', 'No Stop', 'No Stop'] * 2,
        'VALUE_LEVEL': ['L', 'H', 'L', 'H'] * 2,
        'BIDDING_LEVEL': [1, 2, 3, 4, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    return tmp_path

def test_perform_rm_anova_creates_output(sample_data):
    """Test that perform_rm_anova returns a valid AnovaRM result object"""
    result = perform_rm_anova(sample_data)

    # Verify we get the right type of result
    assert result is not None
    assert hasattr(result, 'anova_table')

    # Check ANOVA table has expected effects
    assert "STOP_CONDITION" in result.anova_table.index
    assert "VALUE_LEVEL" in result.anova_table.index
    assert "STOP_CONDITION:VALUE_LEVEL" in result.anova_table.index

def test_perform_rm_anova_categorical_conversion(sample_data):
    """Test that variables are properly converted to categorical"""
    # Make copy to verify conversion
    test_data = sample_data.copy()
    _ = perform_rm_anova(test_data)

    assert test_data["STOP_CONDITION"].dtype.name == "category"
    assert test_data["VALUE_LEVEL"].dtype.name == "category"

def test_perform_rm_anova_results(sample_data):
    """Test that the ANOVA results contain expected columns"""
    result = perform_rm_anova(sample_data)

    # Check that the ANOVA table has the expected columns
    expected_columns = ['F Value', 'Pr > F']
    for col in expected_columns:
        assert col in result.anova_table.columns

    # Check that p-values are floats between 0 and 1
    for p_value in result.anova_table['Pr > F']:
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
