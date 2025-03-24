import pytest

from inhibition_induced_devaluation.utils import calculate_bmi


def test_calculate_bmi():
    """Test that the calculate_bmi function returns the correct BMI value."""
    assert calculate_bmi(70, 1.75) == 22.86


def test_calculate_bmi_negative_values():
    """Test that the calculate_bmi function raises a ValueError
    when the weight or height is negative."""
    with pytest.raises(ValueError):
        calculate_bmi(-70, 1.75)
    with pytest.raises(ValueError):
        calculate_bmi(70, -1.75)
