import numpy as np
import pandas as pd

from inhibition_induced_devaluation.utils.utils import calculate_mean_rts


def test_calculate_mean_rts():
    def test_calculate_mean_rts_basic():
        """Test calculate_mean_rts with simple values"""
        # Create sample DataFrames with known values
        stop_failure = pd.DataFrame({'reaction_time': [100, 200, 300]})
        no_stop_all = pd.DataFrame({'reaction_time': [150, 250, 350]})
        no_stop_go = pd.DataFrame({'reaction_time': [125, 225, 325]})
        no_stop_stop = pd.DataFrame({'reaction_time': [175, 275, 375]})

        # Calculate means
        p2_go_rt, p2_stopfail_rt, p2_gort_go, p2_gort_stop = calculate_mean_rts(
            stop_failure,
            no_stop_all,
            no_stop_go,
            no_stop_stop
        )

        # Check results
        assert p2_go_rt == 250  # mean of [150, 250, 350]
        assert p2_stopfail_rt == 200  # mean of [100, 200, 300]
        assert p2_gort_go == 225  # mean of [125, 225, 325]
        assert p2_gort_stop == 275  # mean of [175, 275, 375]

    def test_calculate_mean_rts_empty():
        """Test calculate_mean_rts with empty DataFrames"""
        empty_df = pd.DataFrame({'reaction_time': []})

        # Calculate means with empty DataFrames
        p2_go_rt, p2_stopfail_rt, p2_gort_go, p2_gort_stop = calculate_mean_rts(
            empty_df,
            empty_df,
            empty_df,
            empty_df
        )

        # Check that empty DataFrames return NaN
        assert np.isnan(p2_go_rt)
        assert np.isnan(p2_stopfail_rt)
        assert np.isnan(p2_gort_go)
        assert np.isnan(p2_gort_stop)

    def test_calculate_mean_rts_single_value():
        """Test calculate_mean_rts with single values"""
        # Create DataFrames with single values
        stop_failure = pd.DataFrame({'reaction_time': [100]})
        no_stop_all = pd.DataFrame({'reaction_time': [150]})
        no_stop_go = pd.DataFrame({'reaction_time': [125]})
        no_stop_stop = pd.DataFrame({'reaction_time': [175]})

        # Calculate means
        p2_go_rt, p2_stopfail_rt, p2_gort_go, p2_gort_stop = calculate_mean_rts(
            stop_failure,
            no_stop_all,
            no_stop_go,
            no_stop_stop
        )

        # Check results
        assert p2_go_rt == 150
        assert p2_stopfail_rt == 100
        assert p2_gort_go == 125
        assert p2_gort_stop == 175

    def test_calculate_mean_rts_with_nan():
        """Test calculate_mean_rts with NaN values"""
        # Create DataFrames with NaN values
        stop_failure = pd.DataFrame({'reaction_time': [100, np.nan, 300]})
        no_stop_all = pd.DataFrame({'reaction_time': [150, 250, np.nan]})
        no_stop_go = pd.DataFrame({'reaction_time': [np.nan, 225, 325]})
        no_stop_stop = pd.DataFrame({'reaction_time': [175, np.nan, 375]})

        # Calculate means
        p2_go_rt, p2_stopfail_rt, p2_gort_go, p2_gort_stop = calculate_mean_rts(
            stop_failure,
            no_stop_all,
            no_stop_go,
            no_stop_stop
        )

        # Check results (means should exclude NaN values)
        assert p2_go_rt == 200  # mean of [150, 250]
        assert p2_stopfail_rt == 200  # mean of [100, 300]
        assert p2_gort_go == 275  # mean of [225, 325]
        assert p2_gort_stop == 275  # mean of [175, 375]
