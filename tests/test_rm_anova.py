import pandas as pd
import pytest
from pathlib import Path
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

def test_perform_rm_anova_creates_output(sample_data, temp_output_dir):
    """Test that perform_rm_anova creates output file with expected content"""
    perform_rm_anova(sample_data, "TEST", "all", temp_output_dir)
    
    # Check if anovas directory was created
    anova_dir = temp_output_dir / "anovas"
    assert anova_dir.exists()
    
    # Check if output file was created
    output_file = anova_dir / "TEST_all_rm_anova_results.txt"
    assert output_file.exists()
    
    # Check file content
    content = output_file.read_text()
    assert "Repeated Measures ANOVA Results for TEST - all subjects" in content
    assert "STOP_CONDITION" in content
    assert "VALUE_LEVEL" in content

def test_perform_rm_anova_categorical_conversion(sample_data, temp_output_dir):
    """Test that variables are properly converted to categorical"""
    # Make copy to verify conversion
    test_data = sample_data.copy()
    perform_rm_anova(test_data, "TEST", "all", temp_output_dir)
    
    assert test_data["STOP_CONDITION"].dtype.name == "category"
    assert test_data["VALUE_LEVEL"].dtype.name == "category"