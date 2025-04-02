import numpy as np

from inhibition_induced_devaluation.utils.utils import calculate_ssrt_value


def test_calculate_ssrt_basic():
    """Test SSRT calculation with simple values"""
    nth_rt = 500
    avg_ssd = 200

    ssrt = calculate_ssrt_value(nth_rt, avg_ssd)
    assert ssrt == 300  # 500 - 200

def test_calculate_ssrt_with_nan():
    """Test SSRT calculation with NaN values"""
    # Test with NaN nth_rt
    ssrt = calculate_ssrt_value(np.nan, 200)
    assert np.isnan(ssrt)

    # Test with NaN avg_ssd
    ssrt = calculate_ssrt_value(500, np.nan)
    assert np.isnan(ssrt)

    # Test with both NaN
    ssrt = calculate_ssrt_value(np.nan, np.nan)
    assert np.isnan(ssrt)

def test_calculate_ssrt_zero_ssd():
    """Test SSRT calculation with zero SSD"""
    ssrt = calculate_ssrt_value(500, 0)
    assert ssrt == 500

def test_calculate_ssrt_negative():
    """Test SSRT calculation resulting in negative value"""
    ssrt = calculate_ssrt_value(200, 300)
    assert ssrt == -100  # Should allow negative values
