# #train_with_simulation.py(for local host)

# """
# train_with_simulation_fixed.py
# Robust training + simulation pipeline for the Dropout prototype.

# What it does:
# - Loads features_cleaned.csv and target_cleaned.csv (created by your extractor)
# - If target has no positives, creates realistic simulated positives using a safe method
# - Preprocesses features (drops ID columns, log1p income, scales numeric)
# - Handles class imbalance safely (SMOTE only when possible and appropriate)
# - Trains an XGBoost classifier with sensible defaults
# - Chooses a threshold robustly (falls back to 0.5 when needed)
# - Prints metrics and saves SHAP mean-abs importances
# """

# import os
# import numpy as np
# import pandas as pd
# import warnings
# import mysql.connector
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
# from xgboost import XGBClassifier

# # optional SMOTE
# try:
#     from imblearn.over_sampling import SMOTE
#     HAS_SMOTE = True
# except Exception:
#     HAS_SMOTE = False

# # SHAP import (wrap in try because shap can be heavy)
# try:
#     import shap
#     HAS_SHAP = True
# except Exception:
#     HAS_SHAP = False

# # ---------------- CONFIG ----------------
# FEATURES_CSV = "features_cleaned.csv"
# TARGET_CSV = "target_cleaned.csv"
# SIM_METHOD = "smart_and"    # options: "smart_and", "smart_or", "random"
# SIM_POSITIVE_FRAC = 0.05   # target fraction of positives if fallback required
# SMART_PERC = 0.10          # percentile used by smart rules (10th percentile)
# RANDOM_SEED = 42
# OUT_SHAP_CSV = "shap_feature_importances_fixed.csv"
# # ----------------------------------------

# rng = np.random.RandomState(RANDOM_SEED)

# def load_data():
#     X = pd.read_csv(FEATURES_CSV, index_col=0)
#     y = pd.read_csv(TARGET_CSV, index_col=0)['dropped'].reindex(X.index).fillna(0).astype(int)
#     return X, y

# def simulate_labels_safely(X, y):
#     """
#     Create simulated positives only when there are zero positives.
#     Strategy:
#       1) Try SMART rule: pick students in bottom SMART_PERC for BOTH attendance and grade (AND).
#       2) If that yields 0 positives or yields all positives, try OR rule.
#       3) If still bad, build a combined percentile score and pick bottom k (SIM_POSITIVE_FRAC).
#       4) If still degenerate, pick random k.
#     Always ensure result has at least one positive and one negative.
#     """
#     if y.sum() > 0:
#         print("Real positives detected: skipping simulation.")
#         return y

#     n = len(X)
#     a = X.get('attendance_rate', pd.Series(0, index=X.index)).fillna(0)
#     g = X.get('avg_grade', pd.Series(0, index=X.index)).fillna(0)

#     att_cut = a.quantile(SMART_PERC)
#     grade_cut = g.quantile(SMART_PERC)

#     # 1) strict AND rule
#     simulated = ((a <= att_cut) & (g <= grade_cut)).astype(int)
#     print(f"Attempted SMART_AND: attendance <= p{int(100*SMART_PERC)} AND avg_grade <= p{int(100*SMART_PERC)} -> {simulated.sum()} positives")

#     # 2) if no positives or all positives, try OR (less strict)
#     if simulated.sum() == 0 or simulated.sum() == n:
#         simulated_or = ((a <= att_cut) | (g <= grade_cut)).astype(int)
#         print(f"Attempted SMART_OR -> {simulated_or.sum()} positives")
#         if 0 < simulated_or.sum() < n:
#             simulated = simulated_or

#     # 3) if still degenerate, pick bottom combined percentile score to get target fraction
#     if simulated.sum() == 0 or simulated.sum() == n:
#         # combined score: lower attendance and lower grade => lower score
#         # we use percentile ranks (0..1) and take average rank
#         att_rank = a.rank(method='average', pct=True)
#         grade_rank = g.rank(method='average', pct=True)
#         combined = (att_rank + grade_rank) / 2.0
#         k = max(1, int(n * SIM_POSITIVE_FRAC))
#         bottom_idx = combined.nsmallest(k).index
#         simulated = pd.Series(0, index=X.index)
#         simulated.loc[bottom_idx] = 1
#         print(f"Fallback: selected bottom {k} by combined percentile score -> {simulated.sum()} positives")

#     # 4) final safety: ensure both classes exist
#     if simulated.nunique() == 1:
#         # create random small set
#         k = max(1, int(n * SIM_POSITIVE_FRAC))
#         idx = rng.choice(X.index, size=k, replace=False)
#         simulated = pd.Series(0, index=X.index)
#         simulated.loc[idx] = 1
#         print(f"Final fallback: randomly selected {k} positives")

#     print("Final simulated positives:", simulated.sum())
#     return simulated.astype(int)

# def preprocess_features(X):
#     # drop ID-like columns
#     for c in ['student_pk', 'student_id', 'studentid', 'id']:
#         if c in X.columns:
#             X = X.drop(columns=[c])

#     # log1p household_income to reduce dominance
#     if 'household_income' in X.columns:
#         X['household_income'] = pd.to_numeric(X['household_income'], errors='coerce').fillna(0)
#         X['household_income'] = np.log1p(X['household_income'])

#     # ensure numeric and fill NaNs
#     X_num = X.select_dtypes(include=[np.number]).copy()
#     # if there are no numeric columns, raise
#     if X_num.shape[1] == 0:
#         raise RuntimeError("No numeric features found after dropping IDs. Check your features file.")
#     X_num = X_num.fillna(0)

#     # scale
#     scaler = StandardScaler()
#     X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
#     return X_scaled, scaler

# def safe_smote_resample(X_train, y_train):
#     if not HAS_SMOTE:
#         return X_train, y_train, False
#     # if only one positive or negative, SMOTE will fail: check counts
#     pos = int((y_train==1).sum())
#     neg = int((y_train==0).sum())
#     if pos < 2 or neg < 2:
#         return X_train, y_train, False
#     sm = SMOTE(random_state=RANDOM_SEED)
#     Xr, yr = sm.fit_resample(X_train, y_train)
#     return Xr, yr, True

# def choose_threshold(y_true, y_prob, min_precision=0.1):
#     # Returns threshold chosen to try to maximize recall subject to min_precision; fallback 0.5
#     try:
#         if len(np.unique(y_true)) < 2:
#             return 0.5
#         precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
#         # find thresholds where precision >= min_precision
#         mask = precisions >= min_precision
#         if mask.any():
#             # choose threshold with max recall among those
#             valid_recalls = recalls[mask[:-1]]  # precisions has one extra element; align properly
#             valid_thresholds = thresholds[mask[:-1]]
#             if len(valid_thresholds) > 0:
#                 best_idx = np.nanargmax(valid_recalls)
#                 return float(valid_thresholds[best_idx])
#         # fallback to threshold that gives max F1
#         f1s = (2 * precisions[:-1] * recalls) / (precisions[:-1] + recalls + 1e-12)
#         best_f1_idx = np.nanargmax(f1s)
#         return float(thresholds[best_f1_idx])
#     except Exception:
#         return 0.5

# def train_and_report(X, y):
#     # ensure both classes
#     if len(np.unique(y)) < 2:
#         print("ERROR: only one class present in labels. Cannot train supervised model.")
#         return

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED)
#     print("Train pos/neg:", int((y_train==1).sum()), int((y_train==0).sum()))

#     # Try SMOTE safely
#     X_train_res, y_train_res, used_smote = safe_smote_resample(X_train, y_train)
#     if used_smote:
#         print("SMOTE applied to training data.")
#     else:
#         print("SMOTE not applied (missing or insufficient minority samples). Using scale_pos_weight if needed.")

#     # compute scale_pos_weight if training on original imbalanced data
#     if used_smote:
#         scale_pos_weight = 1.0
#     else:
#         pos = max(1, int((y_train==1).sum()))
#         neg = max(1, int((y_train==0).sum()))
#         scale_pos_weight = neg / pos

#     model = XGBClassifier(
#         n_estimators=200,
#         max_depth=4,
#         learning_rate=0.05,
#         use_label_encoder=False,
#         eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight,
#         random_state=RANDOM_SEED
#     )
#     model.fit(X_train_res, y_train_res)

#     # predict
#     y_prob = model.predict_proba(X_test)[:,1]
#     thresh = choose_threshold(y_test.values, y_prob, min_precision=0.1)
#     y_pred = (y_prob >= thresh).astype(int)
#     print(f"Chosen threshold = {thresh:.3f}")

#     print("\nClassification report:")
#     print(classification_report(y_test, y_pred, zero_division=0))
#     try:
#         print("ROC AUC:", roc_auc_score(y_test, y_prob))
#     except Exception:
#         print("ROC AUC: could not compute (probable single class in test)")

#     print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

#     # SHAP (if available) — compute mean abs shap values
#     try:
#         if HAS_SHAP:
#             explainer = shap.Explainer(model)
#             shap_vals = explainer(X_test)
#             mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
#             feat_imp = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
#             print("\nTop features by mean(|SHAP|):")
#             print(feat_imp.head(15))
#             feat_imp.to_csv(OUT_SHAP_CSV)
#             print(f"Saved SHAP importances -> {OUT_SHAP_CSV}")
#         else:
#             print("SHAP not installed — skipping SHAP explanations.")
#     except Exception as e:
#         print("SHAP failed:", str(e))
#     return model, thresh

# def main():
#     print("Loading data...")
#     X, y = load_data()
#     print("Dataset loaded:", X.shape, "label counts:", y.value_counts().to_dict())

#     if y.sum() == 0:
#         print("No positive labels found -> simulating labels safely.")
#         y = simulate_labels_safely(X, y)
#         print("Simulated labels distribution:", y.value_counts().to_dict())

#     # Preprocess
#     X_proc, scaler = preprocess_features(X)

#     # Train and get model + threshold
#     model, thresh = train_and_report(X_proc, y)

#     # === Make predictions for all students ===
#     final_probs = model.predict_proba(X_proc)[:, 1]
#     final_preds = (final_probs >= thresh).astype(int)

#     df_predictions = X.copy()  # keep original features (student_id, name, etc.)
#     df_predictions['predicted_label'] = final_preds
#     df_predictions['dropout_prob'] = final_probs

#     # === Filter only at-risk students ===
#     df_risk = df_predictions[df_predictions['predicted_label'] == 1]

#     if df_risk.empty:
#         print("No at-risk students predicted. DB not updated.")
#         return

#     # Keep only relevant columns for dropout_master
#     columns_needed = ['student_id', 'student_name', 'failed_courses', 'attendance_count',
#                       'attendance_rate', 'avg_grade', 'predicted_label', 'dropout_prob']
#     for col in columns_needed:
#         if col not in df_risk.columns:
#             df_risk[col] = 0 if col != 'student_name' else 'Unknown'

#     df_risk_to_db = df_risk[columns_needed]

#     # === Update MySQL dropout_master table ===
#     import mysql.connector
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="baha1528",
#         database="MiniProj"
#     )
#     cursor = conn.cursor()

#     for _, row in df_risk_to_db.iterrows():
#         sql = """
#         INSERT INTO dropout_master
#         (student_id, student_name, failed_courses, attendance_count, attendance_rate, avg_grade, predicted_label, dropout_prob)
#         VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
#         ON DUPLICATE KEY UPDATE
#             failed_courses=VALUES(failed_courses),
#             attendance_count=VALUES(attendance_count),
#             attendance_rate=VALUES(attendance_rate),
#             avg_grade=VALUES(avg_grade),
#             predicted_label=VALUES(predicted_label),
#             dropout_prob=VALUES(dropout_prob);
#         """
#         cursor.execute(sql, (
#             row["student_id"],
#             row["student_name"],
#             row["failed_courses"],
#             row["attendance_count"],
#             row["attendance_rate"],
#             row["avg_grade"],
#             row["predicted_label"],
#             row["dropout_prob"]
#         ))

#     conn.commit()
#     cursor.close()
#     conn.close()
#     print(f"✅ {len(df_risk_to_db)} at-risk students inserted/updated in dropout_master")

#     # Optional: save predictions CSV for reference
#     df_predictions.to_csv("predictions.csv", index=False)
#     print("✅ Saved predictions.csv")


# if __name__ == "__main__":
#     main()


#train_with_simulation.py (for collab)
"""
train_with_simulation_colab.py
Colab-friendly training pipeline for Dropout Prediction.

✔ Loads features_cleaned.csv and target_cleaned.csv
✔ Handles class imbalance (SMOTE)
✔ Trains XGBoost classifier
✔ Evaluates using Accuracy, Precision, Recall, F1, and AUC
✔ Saves predictions.csv locally
🚫 Skips MySQL and SHAP (to work smoothly on Colab)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ---------------- CONFIG ----------------
FEATURES_CSV = "features_cleaned.csv"
TARGET_CSV = "target_cleaned.csv"
SIM_POSITIVE_FRAC = 0.05
SMART_PERC = 0.10
RANDOM_SEED = 42
# ----------------------------------------

rng = np.random.RandomState(RANDOM_SEED)

def load_data():
    X = pd.read_csv(FEATURES_CSV, index_col=0)
    y = pd.read_csv(TARGET_CSV, index_col=0)['dropped'].reindex(X.index).fillna(0).astype(int)
    return X, y

def preprocess_features(X):
    for c in ['student_pk','student_id','studentid','id']:
        if c in X.columns:
            X = X.drop(columns=[c])
    if 'household_income' in X.columns:
        X['household_income'] = np.log1p(pd.to_numeric(X['household_income'], errors='coerce').fillna(0))
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
    return X_scaled

def safe_smote_resample(X_train, y_train):
    if not HAS_SMOTE: return X_train, y_train, False
    pos = (y_train==1).sum(); neg = (y_train==0).sum()
    if pos < 2 or neg < 2: return X_train, y_train, False
    sm = SMOTE(random_state=RANDOM_SEED)
    Xr, yr = sm.fit_resample(X_train, y_train)
    return Xr, yr, True

def choose_threshold(y_true, y_prob, min_precision=0.1):
    """
    Choose an optimal threshold:
    - Prefer max recall while maintaining min precision
    - Fallback to best F1-score
    - Safe against shape mismatch errors
    """
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # Match lengths properly
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        mask = precisions >= min_precision
        if mask.any():
            valid_recalls = recalls[mask]
            valid_thresholds = thresholds[mask]
            if len(valid_thresholds) > 0:
                best_idx = np.nanargmax(valid_recalls)
                return float(valid_thresholds[best_idx])

        # Fallback to threshold that gives best F1-score
        f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
        best_f1_idx = np.nanargmax(f1s)
        return float(thresholds[best_f1_idx])
    except Exception as e:
        print("⚠️ Threshold selection failed, defaulting to 0.5:", e)
        return 0.5


def train_and_report(X, y):
    if len(np.unique(y)) < 2:
        print("❌ Only one class found — cannot train supervised model.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED)
    print("Train pos/neg:", int((y_train==1).sum()), int((y_train==0).sum()))
    X_train_res, y_train_res, used_smote = safe_smote_resample(X_train, y_train)
    print("SMOTE applied." if used_smote else "SMOTE skipped.")
    pos = max(1,(y_train==1).sum()); neg = max(1,(y_train==0).sum())
    scale_pos_weight = 1.0 if used_smote else neg/pos
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric='logloss', scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED, use_label_encoder=False
    )
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:,1]
    thresh = choose_threshold(y_test.values, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    print(f"Chosen threshold = {thresh:.3f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        print("ROC AUC could not be computed.")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return model, thresh

def main():
    print("Loading data...")
    X, y = load_data()
    print("Dataset loaded:", X.shape, "label counts:", y.value_counts().to_dict())
    X_proc = preprocess_features(X)
    model, thresh = train_and_report(X_proc, y)
    if model is None:
        return
    final_probs = model.predict_proba(X_proc)[:,1]
    final_preds = (final_probs >= thresh).astype(int)
    df_predictions = X.copy()
    df_predictions['predicted_label'] = final_preds
    df_predictions['dropout_prob'] = final_probs
    df_predictions.to_csv("predictions.csv", index=False)
    print("\n✅ Saved predictions.csv successfully.")
    print("First few rows of predictions:\n", df_predictions.head())

    # === Risk Summary ===
    at_risk = df_final[df_final["predicted_label"] == 1].copy()
    num_at_risk = len(at_risk)
    
    print(f"\n⚠️ {num_at_risk} students are predicted to be at risk of dropping out.")
    
    if num_at_risk > 0:
        # Sort by highest dropout probability
        at_risk = at_risk.sort_values(by="dropout_prob", ascending=False)
        print("\nTop at-risk students:")
        display_cols = ["student_id", "avg_grade", "attendance_rate", "failed_courses", "dropout_prob"]
        # Show top 10 or all if fewer
        print(at_risk[display_cols].head(10).to_string(index=False))
    else:
        print("✅ Great! No students are predicted to be at risk.")


if __name__ == "__main__":
    main()








