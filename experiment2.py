# ... existing code ...

# Replace the existing feedback logic with:
        # Check if we should show feedback
        if should_show_feedback(count, practice_len, first_block_length if count < practice_len + first_block_length else second_block_length):
            # Show feedback based on last 12 trials
            show_feedback_and_break(
                mywin=mywin,
                feedback_tracker=feedback_tracker[-12:],  # Only use last 12 trials
                break_feedback=break_feedback,
                practice_feedback_late=practice_feedback_late,
                practice_feedback_pressure=practice_feedback_pressure,
                practice_feedback_hit=practice_feedback_hit,
                practice_feedback_depress=practice_feedback_depress,
                practice_feedback_countdown=practice_feedback_countdown,
                num_trials=12,  # Always based on 12 trials
                break_time=BREAK_TIME
            )
            # Reset feedback tracker after showing feedback
            feedback_tracker = []

# ... existing code ... 