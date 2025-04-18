import pandas as pd

from inhibition_induced_devaluation.utils.utils import (
    calculate_iid_effect,
    calculate_mean_bids,
    get_phase3_data,
    separate_shape_types,
)


def test_get_phase3_data():
    """Test extraction of phase 3 data"""
    data = {
        'which_phase': ['phase_1', 'phase_2', 'phase_3', 'phase_3'],
        'value': [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)

    phase3_df = get_phase3_data(df)
    assert len(phase3_df) == 2
    assert all(phase3_df['which_phase'] == 'phase_3')

def test_separate_shape_types():
    """Test separation of stop and go shapes"""
    data = {
        'paired_with_stopping': [1, 1, 0, 0],
        'value': [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)

    stop_shapes, go_shapes = separate_shape_types(df)
    assert len(stop_shapes) == 2
    assert len(go_shapes) == 2
    assert all(stop_shapes['paired_with_stopping'] == 1)
    assert all(go_shapes['paired_with_stopping'] == 0)

def test_calculate_mean_bids():
    """Test calculation of mean bidding levels"""
    stop_shapes = pd.DataFrame({
        'chosen_bidding_level': [1, 3]
    })
    go_shapes = pd.DataFrame({
        'chosen_bidding_level': [2, 4]
    })

    stop_bid, go_bid = calculate_mean_bids(stop_shapes, go_shapes)
    assert stop_bid == 2  # (1 + 3) / 2
    assert go_bid == 3    # (2 + 4) / 2

def test_calculate_iid_effect():
    """Test calculation of IID effect"""
    # Positive IID effect (devaluation)
    effect1 = calculate_iid_effect(stop_bid=2, go_bid=3)
    assert effect1 == 1

    # Negative IID effect (no devaluation)
    effect2 = calculate_iid_effect(stop_bid=3, go_bid=2)
    assert effect2 == -1

    # No effect
    effect3 = calculate_iid_effect(stop_bid=2, go_bid=2)
    assert effect3 == 0
