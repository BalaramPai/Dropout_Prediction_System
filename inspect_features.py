# inspect_features.py
import pandas as pd

X = pd.read_csv("features_cleaned.csv", index_col=0)
print("SHAPE:", X.shape)
print("\nCOLUMNS:\n", X.columns.tolist())
print("\nNULLs per column:\n", X.isna().sum())
print("\nUNIQUE counts (sorted):\n", X.nunique().sort_values().head(40))
print("\nSUMMARY for key columns:")
for c in ['avg_grade','attendance_rate','failed_courses','grade_trend','total_participation','household_income']:
    if c in X.columns:
        print(f"\n== {c} ==")
        print(X[c].describe())
        print("Top values:\n", X[c].value_counts().head(10))
