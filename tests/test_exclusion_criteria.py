import numpy as np

from inhibition_induced_devaluation.utils.utils import check_exclusion_criteria


def test_check_exclusion_criteria_include():
    """Test when subject should be included (passes both criteria)"""
    # Stop-fail RT < No-stop RT and SSRT > min cutoff
    subject_vector, reason = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=150,
        min_ssrt_cutoff=100
    )

    assert subject_vector == [1, 1]
    assert reason == "include - subject passed all criteria"

def test_check_exclusion_criteria_exclude_ssrt():
    """Test exclusion due to SSRT below cutoff"""
    # SSRT below minimum cutoff
    subject_vector, reason = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=50,  # Below min_ssrt_cutoff
        min_ssrt_cutoff=100
    )

    assert subject_vector == [1, 0]
    assert reason == "exclude - SSRT is lower than the minimum SSRT"

def test_check_exclusion_criteria_exclude_rt():
    """Test exclusion due to stop-fail RT >= no-stop RT"""
    # Stop-fail RT >= No-stop RT
    subject_vector, reason = check_exclusion_criteria(
        mean_rt_stop_fail=500,  # Equal to no-stop RT
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=150,
        min_ssrt_cutoff=100
    )

    assert subject_vector == [0, 1]
    assert reason == "exclude - stop fail RT is >= no-stop RT"

def test_check_exclusion_criteria_exclude_both():
    """Test exclusion due to failing both criteria"""
    subject_vector, reason = check_exclusion_criteria(
        mean_rt_stop_fail=500,
        mean_rt_no_stop_trials_stop_shapes=400,
        ssrt=50,
        min_ssrt_cutoff=100
    )

    assert subject_vector == [0, 0]
    assert reason == "exclude - failed both criteria"

def test_check_exclusion_criteria_nan_ssrt():
    """Test with NaN SSRT value"""
    subject_vector, reason = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=np.nan,
        min_ssrt_cutoff=100
    )

    assert subject_vector == [1, 0]
    assert reason == "exclude - SSRT is lower than the minimum SSRT"

def test_check_exclusion_criteria_custom_cutoff():
    """Test with custom SSRT cutoff value"""
    # Should pass with lower cutoff
    subject_vector1, reason1 = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=80,
        min_ssrt_cutoff=50
    )
    assert subject_vector1 == [1, 1]
    assert reason1 == "include - subject passed all criteria"

    # Should fail with higher cutoff
    subject_vector2, reason2 = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=80,
        min_ssrt_cutoff=150
    )
    assert subject_vector2 == [1, 0]
    assert reason2 == "exclude - SSRT is lower than the minimum SSRT"

def test_check_exclusion_criteria_edge_cases():
    """Test edge cases with very close values"""
    # Test when stop-fail RT is just slightly less than no-stop RT
    subject_vector1, reason1 = check_exclusion_criteria(
        mean_rt_stop_fail=499.99,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=150,
        min_ssrt_cutoff=100
    )
    assert subject_vector1 == [1, 1]

    # Test when SSRT is exactly at cutoff
    subject_vector2, reason2 = check_exclusion_criteria(
        mean_rt_stop_fail=400,
        mean_rt_no_stop_trials_stop_shapes=500,
        ssrt=100,
        min_ssrt_cutoff=100
    )
    assert subject_vector2 == [1, 1]
