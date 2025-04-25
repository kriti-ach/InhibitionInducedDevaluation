import numpy as np
import pandas as pd

from inhibition_induced_devaluation.utils.phase2_utils import calculate_nth_rt


def test_calculate_nth_rt_basic():
    """Test nth RT calculation with simple values"""
    data = {
        'reaction_time': [100, 200, 300, 400, 500]
    }
    df = pd.DataFrame(data)

    # Test getting the 2nd RT (index 1)
    nth_rt = calculate_nth_rt(df, 1)
    assert nth_rt == 200

def test_calculate_nth_rt_with_zeros():
    """Test nth RT calculation with zero RTs that should be replaced"""
    data = {
        'reaction_time': [0, 100, 200, 0, 300]
    }
    df = pd.DataFrame(data)

    # The zeros should be replaced with 1000
    # So ordering should be: 100, 200, 300, 1000, 1000
    nth_rt = calculate_nth_rt(df, 3)
    assert nth_rt == 1000

def test_calculate_nth_rt_out_of_range():
    """Test when rank is out of range"""
    data = {
        'reaction_time': [100, 200, 300]
    }
    df = pd.DataFrame(data)

    # Test with rank larger than DataFrame
    nth_rt = calculate_nth_rt(df, 5)
    assert np.isnan(nth_rt)

def test_calculate_nth_rt_empty():
    """Test with empty DataFrame"""
    df = pd.DataFrame({'reaction_time': []})
    nth_rt = calculate_nth_rt(df, 0)
    assert np.isnan(nth_rt)
