"""Quick test to debug credit mapping."""
import pandas as pd
import numpy as np

# Load existing data
print("Loading data...")
seq_df = pd.read_csv('outputs/modeling/cxa_v2/action_sequences.csv')
act_df = pd.read_csv('outputs/modeling/cxa_v2/actions.csv')

print(f"Loaded {len(seq_df)} sequences and {len(act_df)} actions")
print()

# Check sequence IDs
print("Sequence ID column checks:")
print(f"  In action_df: {'sequence_id' in act_df.columns}")
print(f"  In sequence_df: {'sequence_id' in seq_df.columns}")
print()

# Sample IDs
print("Sample sequence IDs:")
print(f"  From action_df: {act_df['sequence_id'].head().tolist()}")
print(f"  From sequence_df: {seq_df['sequence_id'].head().tolist()}")
print()

# Check if they match
act_seq_ids = set(act_df['sequence_id'].unique())
seq_seq_ids = set(seq_df['sequence_id'].unique())
print(f"Unique sequence IDs in action_df: {len(act_seq_ids)}")
print(f"Unique sequence IDs in sequence_df: {len(seq_seq_ids)}")
print(f"IDs match: {act_seq_ids == seq_seq_ids}")
print()

# Simulate credit distribution
print("Simulating credit distribution...")
# Create fake credit values
seq_df_with_credit = seq_df.copy()
for pos in range(1, 6):
    seq_df_with_credit[f"action{pos}_credit"] = np.random.uniform(0, 0.5, len(seq_df))

print(f"Credit columns added: {[c for c in seq_df_with_credit.columns if 'credit' in c][:3]}")
print()

# Test mapping
print("Testing credit mapping...")
act_df_test = act_df.copy()
act_df_test["credit"] = 0.0

for pos in range(1, 6):
    credit_col = f"action{pos}_credit"
    pos_mask = act_df_test["action_position"] == pos
    
    # Create mapping
    credit_map = seq_df_with_credit.set_index("sequence_id")[credit_col].to_dict()
    
    # Apply mapping
    act_df_test.loc[pos_mask, "credit"] = act_df_test.loc[pos_mask, "sequence_id"].map(credit_map)
    
    mapped_count = act_df_test.loc[pos_mask, "credit"].notna().sum()
    nonzero_count = (act_df_test.loc[pos_mask, "credit"] > 0).sum()
    print(f"  Position {pos}: {pos_mask.sum()} actions, {mapped_count} mapped, {nonzero_count} non-zero")

print()
print("Final credit stats:")
print(act_df_test["credit"].describe())
print(f"Non-zero credits: {(act_df_test['credit'] > 0).sum()}")
