# Example of a well-documented function
def calculate_bmi(weight: float, height: float) -> float:
    """Calculate Body Mass Index (BMI) given weight and height.

    Args:
        weight (float): The weight of the individual in kilograms (kg).
        height (float): The height of the individual in meters (m).

    Raises:
        ValueError: If either weight or height is non-positive.

    Returns:
        float: The calculated BMI value, rounded to two decimal places.
    """
    if weight <= 0 or height <= 0:
        raise ValueError('Weight and height must be positive values.')

    bmi = weight / (height**2)
    return round(bmi, 2)
