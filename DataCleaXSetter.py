# import pandas as pd
# import mysql.connector
# from sklearn.preprocessing import OneHotEncoder
# from sqlalchemy import create_engine
#
# # -------------------------------
# # 1️⃣ Connect to MySQL Database
# # -------------------------------
# # conn = mysql.connector.connect(
# #     host='localhost',
# #     user='root',
# #     password='baha1528',  # Replace with your password
# #     database='MiniProj'
# # )
#
#
#
# engine = create_engine("mysql+mysqlconnector://root:baha1528@localhost/MiniProj")
#
#
# # -------------------------------
# # 2️⃣ Load Tables
# # -------------------------------
# students = pd.read_sql("SELECT * FROM students", engine)
# academic = pd.read_sql("SELECT * FROM academic_records", engine)
# attendance = pd.read_sql("SELECT * FROM attendance", engine)
# behavioral = pd.read_sql("SELECT * FROM behavioral", engine)
# dropout = pd.read_sql("SELECT * FROM dropout_labels", engine)
#
# # Fix the target column
# dropout['dropped'] = dropout['dropped'].replace(127, 0)
# # Extract target
# y = dropout['dropped']
# # Optional: check distribution
# print(y.value_counts())
#
# # -------------------------------
# # 3️⃣ Aggregate Features per Student
# # -------------------------------
#
# # Academic features
# academic_agg = academic.groupby('student_id').agg(
#     avg_grade=('grade', 'mean'),
#     failed_courses=('pass_fail', 'sum'),
#     total_credits=('credits', 'sum')
# ).reset_index()
#
# # Attendance features
# attendance_agg = attendance.groupby('student_id').agg(
#     attendance_rate=('present', 'mean')
# ).reset_index()
#
# # Behavioral features
# behavioral_agg = behavioral.groupby('student_id').agg(
#     total_participation=('participation_score', 'sum'),
#     events_attended=('activity_type', 'count')
# ).reset_index()
#
# # -------------------------------
# # 4️⃣ Merge All Features
# # -------------------------------
# features = students.merge(academic_agg, on='student_id', how='left')\
#                    .merge(attendance_agg, on='student_id', how='left')\
#                    .merge(behavioral_agg, on='student_id', how='left')\
#                    .merge(dropout[['student_id', 'dropped']], on='student_id', how='left')
#
# # -------------------------------
# # 5️⃣ Handle Missing Values
# # -------------------------------
# # Numeric columns: fill missing with 0
# numeric_cols = ['avg_grade', 'failed_courses', 'total_credits', 'attendance_rate',
#                 'total_participation', 'events_attended', 'household_income']
# for col in numeric_cols:
#     features[col] = features[col].fillna(0)
#
# # Categorical columns: fill missing with 'Unknown'
# categorical_cols = ['gender', 'socioeconomic_cat', 'first_gen']
# for col in categorical_cols:
#     features[col] = features[col].fillna('Unknown')
#
# # -------------------------------
# # 6️⃣ Encode Categorical Variables
# # -------------------------------
# encoder = OneHotEncoder(sparse_output=False, drop='first')
# encoded = pd.DataFrame(encoder.fit_transform(features[categorical_cols]),
#                        columns=encoder.get_feature_names_out(categorical_cols))
#
# # Drop original categorical columns and add encoded ones
# features = features.drop(columns=categorical_cols).reset_index(drop=True)
# features = pd.concat([features, encoded], axis=1)
#
# # -------------------------------
# # 7️⃣ Separate Features and Target
# # -------------------------------
# X = features.drop(columns=['student_id', 'full_name', 'dropped', 'join_year', 'created_at'])
# y = features['dropped']
#
# # -------------------------------
# # 8️⃣ Ready to use for ML
# # -------------------------------
# print("Features (X):")
# print(X.head())
# print("\nTarget (y):")
# print(y.head())









# DataCleaXSetter.py
# Robust feature extraction from MySQL for Dropout prediction
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder

# --------- CONFIG ----------
DB_USER = "root"
DB_PASS = "baha1528"           # change if needed
DB_HOST = "localhost"
DB_NAME = "MiniProj"
OUT_FEATURES_CSV = "features_cleaned.csv"
OUT_TARGET_CSV = "target_cleaned.csv"
# ---------------------------

def connect_engine(user, password, host, db):
    uri = f"mysql+mysqlconnector://{user}:{password}@{host}/{db}"
    return create_engine(uri, pool_pre_ping=True)

def latest_label_per_student(dropout_df):
    # If multiple label snapshots exist, take the latest label_date per student
    if 'label_date' in dropout_df.columns:
        # try parse dates
        dropout_df['label_date'] = pd.to_datetime(dropout_df['label_date'], errors='coerce')
        # sort and take last per student_id
        latest = dropout_df.sort_values(['student_id','label_date']).groupby('student_id', as_index=False).last()
        return latest[['student_id','dropped']]
    else:
        # fallback: assume one record per student
        return dropout_df[['student_id','dropped']]

def compute_grade_trend(academic):
    # returns DataFrame with student_id and grade_trend (slope)
    def slope(series_dates, series_grades):
        try:
            if len(series_grades) < 2:
                return 0.0
            # x = ordinal representation of dates if available, else index
            if series_dates.notnull().sum() >= 2:
                x = series_dates.dropna().map(pd.Timestamp.toordinal).values
                y = series_grades.loc[series_dates.notnull()].values
            else:
                x = np.arange(len(series_grades))
                y = series_grades.values
            m = np.polyfit(x, y, 1)[0]
            return float(m)
        except Exception:
            return 0.0

    rows = []
    for sid, g in academic.groupby('student_id'):
        # sort by term_date if possible
        if 'term_date' in g.columns:
            g = g.copy()
            g['term_date'] = pd.to_datetime(g['term_date'], errors='coerce')
            g = g.sort_values('term_date')
        grades = g['grade'].fillna(method='ffill').fillna(0)
        dates = g['term_date'] if 'term_date' in g.columns else pd.Series([None]*len(g))
        rows.append({'student_id': sid, 'grade_trend': slope(dates, grades)})
    return pd.DataFrame(rows)

def main():
    engine = connect_engine(DB_USER, DB_PASS, DB_HOST, DB_NAME)

    # 1) Load tables via SQLAlchemy engine
    students = pd.read_sql("SELECT * FROM students", engine)
    academic = pd.read_sql("SELECT * FROM academic_records", engine)
    attendance = pd.read_sql("SELECT * FROM attendance", engine)
    behavioral = pd.read_sql("SELECT * FROM behavioral", engine)
    dropout = pd.read_sql("SELECT * FROM dropout_labels", engine)

    # 2) Clean target: normalize values and take latest snapshot if multiple
    # Replace placeholder 127 with 0 (not dropped) if present
    if 'dropped' in dropout.columns:
        dropout['dropped'] = pd.to_numeric(dropout['dropped'], errors='coerce').fillna(0).astype(int)
        dropout['dropped'] = dropout['dropped'].replace(127, 0)
    else:
        raise RuntimeError("dropout_labels table doesn't contain 'dropped' column")

    latest_dropout = latest_label_per_student(dropout)
    # Ensure dropped is 0/1
    latest_dropout['dropped'] = latest_dropout['dropped'].apply(lambda v: 1 if int(v) == 1 else 0)

    # 3) Aggregate academic features
    # Ensure numeric types
    for col in ['grade','credits','pass_fail']:
        if col in academic.columns:
            academic[col] = pd.to_numeric(academic[col], errors='coerce').fillna(0)
    # avg grade, total credits
    acad_agg = academic.groupby('student_id').agg(
        avg_grade=('grade', 'mean'),
        total_credits=('credits', 'sum'),
        pass_rate=('pass_fail', 'mean')  # if pass_fail is 1 for pass
    ).reset_index()
    # compute failed_courses = count rows where pass_fail == 0 (and not null)
    failed = academic.assign(failed = (academic['pass_fail'] == 0).astype(int)).groupby('student_id')['failed'].sum().reset_index()
    failed.rename(columns={'failed': 'failed_courses'}, inplace=True)
    acad_agg = acad_agg.merge(failed, on='student_id', how='left')

    # grade trend
    grade_trend_df = compute_grade_trend(academic)
    acad_agg = acad_agg.merge(grade_trend_df, on='student_id', how='left')

    # 4) Aggregate attendance features
    # present should be numeric 0/1
    if 'present' in attendance.columns:
        attendance['present'] = pd.to_numeric(attendance['present'], errors='coerce').fillna(0).astype(float)
    att_agg = attendance.groupby('student_id').agg(
        attendance_rate=('present', 'mean'),
        attendance_count=('present', 'count')
    ).reset_index()

    # 5) Aggregate behavioral features
    if 'participation_score' in behavioral.columns:
        behavioral['participation_score'] = pd.to_numeric(behavioral['participation_score'], errors='coerce').fillna(0)
    behavioral_agg = behavioral.groupby('student_id').agg(
        total_participation=('participation_score', 'sum'),
        events_attended=('activity_type', 'count')
    ).reset_index()

    # 6) Merge features onto students
    features = students.merge(acad_agg, on='student_id', how='left')\
                       .merge(att_agg, on='student_id', how='left')\
                       .merge(behavioral_agg, on='student_id', how='left')\
                       .merge(latest_dropout[['student_id','dropped']], on='student_id', how='left')

    # 7) Basic cleaning: fill NaNs for numeric features with 0
    numeric_fill_cols = ['avg_grade','total_credits','pass_rate','failed_courses','grade_trend',
                         'attendance_rate','attendance_count','total_participation','events_attended','household_income']
    for c in numeric_fill_cols:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c], errors='coerce').fillna(0)

    # 8) Handle categorical features
    # Keep first_gen as numeric (0/1) if present
    if 'first_gen' in features.columns:
        # some CSVs store it as '0'/'1' or 'True'/'False' or text; normalize to 0/1
        features['first_gen'] = pd.to_numeric(features['first_gen'], errors='coerce').fillna(0).astype(int)

    categorical_cols = []
    for col in ['gender','socioeconomic_cat']:
        if col in features.columns:
            features[col] = features[col].fillna('Unknown').astype(str)
            categorical_cols.append(col)

    # One-hot encode categorical columns (if any)
    encoded_df = pd.DataFrame()
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_arr = encoder.fit_transform(features[categorical_cols])
        encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(categorical_cols), index=features.index)

    # 9) Final feature set preparation
    # Drop text/irrelevant columns: full_name, dob, created_at, join_year (optional), student_id kept for join but removed later
    drop_cols_if_exist = ['full_name','dob','created_at']  # keep join_year out of X for now
    for c in drop_cols_if_exist:
        if c in features.columns:
            features = features.drop(columns=[c])

    # concat encoded categorical if any
    if not encoded_df.empty:
        features = pd.concat([features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # columns to drop for X
    to_drop_for_X = ['student_id','dropped','join_year']  # keep join_year optional; remove it here
    X = features.drop(columns=[c for c in to_drop_for_X if c in features.columns])
    y = features['dropped'].fillna(0).astype(int)

    # final check: ensure all X columns numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # coerce to numeric where possible, else fill 0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Reindex X,y by student_id if available for traceability
    if 'student_id' in features.columns:
        X.index = features['student_id']
        y.index = features['student_id']

    # Report summary
    print("---- Feature extraction summary ----")
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts(dropna=False))
    print("Feature example:\n", X.head())
    print("------------------------------------")

    # Save outputs for ML
    X.to_csv(OUT_FEATURES_CSV, index=True)
    y.to_csv(OUT_TARGET_CSV, index=True, header=['dropped'])

    print(f"Saved features -> {OUT_FEATURES_CSV}")
    print(f"Saved target   -> {OUT_TARGET_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()



















