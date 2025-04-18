import numpy as np
import pandas as pd

from inhibition_induced_devaluation.utils.utils import calculate_avg_ssd


def test_calculate_avg_ssd_basic():
    """Test average SSD calculation with simple values"""
    data = {
        'stim_location': [5, 5, 6, 6],
        'left_SSD': [100, 200, np.nan, np.nan],
        'right_SSD': [np.nan, np.nan, 300, 400]
    }
    df = pd.DataFrame(data)

    avg_ssd = calculate_avg_ssd(df)
    assert avg_ssd == 250  # (150 + 350) / 2

def test_calculate_avg_ssd_single_side():
    """Test when only one side has data"""
    data = {
        'stim_location': [5, 5],
        'left_SSD': [100, 200],
        'right_SSD': [np.nan, np.nan]
    }
    df = pd.DataFrame(data)

    avg_ssd = calculate_avg_ssd(df)
    assert np.isnan(avg_ssd)  # Should be NaN when one side is missing

def test_calculate_avg_ssd_empty():
    """Test with empty DataFrame"""
    df = pd.DataFrame({
        'stim_location': [],
        'left_SSD': [],
        'right_SSD': []
    })

    avg_ssd = calculate_avg_ssd(df)
    assert np.isnan(avg_ssd)

def test_calculate_avg_ssd_no_matching_stim_locations():
    """Test when no trials match the stim_location criteria"""
    data = {
        'stim_location': [1, 2, 3, 4],
        'left_SSD': [100, 200, 300, 400],
        'right_SSD': [100, 200, 300, 400]
    }
    df = pd.DataFrame(data)

    avg_ssd = calculate_avg_ssd(df)
    assert np.isnan(avg_ssd)
