import pandas as pd

from inhibition_induced_devaluation.utils.utils import calculate_p_respond


def test_calculate_p_respond_basic():
    """Test p_respond calculation with simple values"""
    # Create sample data where 2 out of 4 trials had responses
    data = {
        'stop_signal_trial_type': ['stop'] * 4,
        'response': [1, 1, 0, 0],
        'accuracy': [1, 1, 0, 0]
    }
    trials_with_ss = pd.DataFrame(data)

    p_respond = calculate_p_respond(trials_with_ss)
    assert p_respond == 0.5  # 2 responses out of 4 trials

def test_calculate_p_respond_all_respond():
    """Test when all trials have responses"""
    data = {
        'stop_signal_trial_type': ['stop'] * 3,
        'response': [1, 1, 1],
        'accuracy': [1, 1, 1]
    }
    trials_with_ss = pd.DataFrame(data)

    p_respond = calculate_p_respond(trials_with_ss)
    assert p_respond == 1.0

def test_calculate_p_respond_none_respond():
    """Test when no trials have responses"""
    data = {
        'stop_signal_trial_type': ['stop'] * 3,
        'response': [0, 0, 0],
        'accuracy': [0, 0, 0]
    }
    trials_with_ss = pd.DataFrame(data)

    p_respond = calculate_p_respond(trials_with_ss)
    assert p_respond == 0.0
