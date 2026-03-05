"""Quick missingness check for passes.csv"""
import pandas as pd

df = pd.read_csv('outputs/analysis/cxa/data/passes.csv')
print(f'Total rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')
print()
print('MISSINGNESS REPORT')
print('=' * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

report = list(zip(missing.index, missing.values, missing_pct.values))
report.sort(key=lambda x: x[2], reverse=True)

for c, m, p in report:
    if m > 0:
        print(f'{c:35} {m:>10,.0f} ({p:>6.2f}%)')

print()
print(f'Complete columns: {sum(1 for _,m,_ in report if m==0)}')
print()
print('SEQUENCE COLUMNS')
print('-' * 60)
seq_cols = ['sequence_id', 'passes_to_shot', 'is_key_pass', 'is_second_assist', 'is_third_assist', 'sequence_xA']
for col in seq_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'{col:35} {non_null:>10,} non-null')
