"""Tests for mapping sequence-level cXA credits onto action rows."""

import pandas as pd


def map_action_credits(seq_df: pd.DataFrame, act_df: pd.DataFrame) -> pd.DataFrame:
    """Map action-position credit columns to matching action rows."""
    mapped = act_df.copy()
    mapped["credit"] = 0.0

    for position in sorted(mapped["action_position"].unique()):
        credit_col = f"action{position}_credit"
        position_mask = mapped["action_position"] == position
        credit_map = seq_df.set_index("sequence_id")[credit_col]
        mapped.loc[position_mask, "credit"] = mapped.loc[position_mask, "sequence_id"].map(
            credit_map
        )

    return mapped


def test_action_credits_map_by_sequence_id_and_action_position() -> None:
    """Each action row receives its sequence's credit for that action position."""
    seq_df = pd.DataFrame(
        {
            "sequence_id": [101, 202],
            "action1_credit": [0.10, 0.40],
            "action2_credit": [0.20, 0.50],
            "action3_credit": [0.30, 0.60],
        }
    )
    act_df = pd.DataFrame(
        {
            "sequence_id": [101, 101, 101, 202, 202, 202],
            "action_position": [1, 2, 3, 1, 2, 3],
        }
    )

    mapped = map_action_credits(seq_df, act_df)

    assert mapped["credit"].tolist() == [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
