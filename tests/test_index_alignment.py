"""Test index alignment in credit distribution."""

import pandas as pd

# Simulate the issue
# Create a dataframe with integer index (like sequence_df)
df = pd.DataFrame(
    {"sequence_id": ["seq1", "seq2", "seq3"], "weight": [0.5, 0.3, 0.2]}, index=[0, 1, 2]
)

# Create a series indexed by sequence_id (like sequence_values)
sequence_value = pd.Series([1.0, 2.0, 3.0], index=["seq1", "seq2", "seq3"])

print("DataFrame:")
print(df)
print()

print("Sequence values (indexed by sequence_id):")
print(sequence_value)
print()

# Try the multiplication
print("Multiplication result (BAD - misaligned indices):")
result_bad = df["weight"] * sequence_value
print(result_bad)
print()

# Correct way - align by matching on sequence_id
print("Correct multiplication (aligned by sequence_id):")
result_good = df["weight"] * df["sequence_id"].map(sequence_value)
print(result_good)
