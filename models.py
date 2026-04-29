"""
models.py — ML training & prediction menggunakan data real dari dvdrental.

3 model dengan feature set berbeda:
  • Logistic Regression  → RFM features (recency, frequency, monetary)
  • Random Forest        → Full behavioral features
  • Gradient Boosting    → Full behavioral + genre diversity
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
)

# ══════════════════════════════════════════════════════════
# FEATURE SETS — setiap model pakai kolom berbeda
# ══════════════════════════════════════════════════════════

FEATURE_SETS = {
    "Logistic Regression": {
        "cols": ["total_rentals", "total_spending", "days_since_last_rental"],
        "description": (
            "**Model RFM (Recency · Frequency · Monetary).** "
            "Hanya 3 fitur: seberapa sering rental, total belanja, dan kapan terakhir "
            "rental. Paling mudah diinterpretasikan oleh bisnis."
        ),
        "color": "#6366f1",
    },
    "Random Forest": {
        "cols": [
            "total_rentals", "avg_days_overdue", "late_return_count",
            "late_return_rate", "total_spending",
            "avg_rental_gap_days", "days_since_last_rental",
        ],
        "description": (
            "**Model Behavioural Lengkap.** Menggunakan 7 fitur termasuk statistik "
            "keterlambatan. Lebih baik mendeteksi pola keterlambatan yang kompleks "
            "karena menggabungkan banyak pohon keputusan (ensemble)."
        ),
        "color": "#10b981",
    },
    "Gradient Boosting": {
        "cols": [
            "total_rentals", "avg_days_overdue", "late_return_count",
            "late_return_rate", "total_spending",
            "avg_rental_gap_days", "days_since_last_rental",
            "genre_diversity",
        ],
        "description": (
            "**Model Extended + Diversitas Genre.** Menambahkan genre diversity — "
            "pelanggan yang mengeksplorasi lebih banyak genre cenderung lebih engaged. "
            "Boosting sekuensial menangkap pola non-linear yang kompleks."
        ),
        "color": "#f59e0b",
    },
}

FEATURE_LABELS = {
    "total_rentals":          "Total Rentals",
    "avg_days_overdue":       "Avg Days Overdue",
    "late_return_count":      "Late Return Count",
    "late_return_rate":       "Late Return Rate",
    "total_spending":         "Total Spending ($)",
    "avg_rental_gap_days":    "Avg Rental Gap (days)",
    "days_since_last_rental": "Days Since Last Rental",
    "genre_diversity":        "Genre Diversity Score",
}


# ══════════════════════════════════════════════════════════
# TRAIN SEMUA MODEL
# ══════════════════════════════════════════════════════════

def train_all_models(risk_df: pd.DataFrame) -> dict:
    """
    Latih 3 model menggunakan data real dari get_risk_profile().
    Return dict: { model_name: { model, scaler, feature_cols, metrics } }
    """
    df  = risk_df.copy().fillna(0)
    y   = df["is_high_risk"].astype(int)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            random_state=42, class_weight="balanced",
            max_iter=1000, C=0.5,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", max_depth=6, min_samples_leaf=3,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=42,
            max_depth=4, min_samples_leaf=3,
            subsample=0.8, learning_rate=0.05,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        cols   = FEATURE_SETS[name]["cols"]
        X      = df[cols].fillna(0)
        scaler = StandardScaler()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=7, stratify=y
        )
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        y_prob = clf.predict_proba(X_te_s)[:, 1]

        cv_res = cross_validate(
            clf, scaler.transform(X), y, cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
        )

        # Feature importance
        if name == "Logistic Regression":
            raw_imp = np.abs(clf.coef_[0])
        else:
            raw_imp = clf.feature_importances_

        fi = {
            FEATURE_LABELS.get(c, c): round(float(v), 4)
            for c, v in zip(cols, raw_imp)
        }

        results[name] = {
            "model":        clf,
            "scaler":       scaler,
            "feature_cols": cols,
            "metrics": {
                "accuracy":         round(accuracy_score(y_te, y_pred),  4),
                "roc_auc":          round(roc_auc_score(y_te, y_prob),   4),
                "f1":               round(f1_score(y_te, y_pred),        4),
                "cv_acc_mean":      round(float(np.mean(cv_res["test_accuracy"])), 4),
                "cv_acc_std":       round(float(np.std(cv_res["test_accuracy"])),  4),
                "cv_auc_mean":      round(float(np.mean(cv_res["test_roc_auc"])),  4),
                "cv_f1_mean":       round(float(np.mean(cv_res["test_f1"])),       4),
                "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
                "report":           classification_report(y_te, y_pred, output_dict=True),
                "feature_importance": fi,
                "train_size":       len(X_tr),
                "test_size":        len(X_te),
            },
        }

    return results


# ══════════════════════════════════════════════════════════
# PREDICT SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════

def predict_customer(
    model_name: str,
    trained_models: dict,
    customer_row: pd.Series,
) -> dict:
    """
    Prediksi risk satu customer.
    customer_row = baris dari risk_profile DataFrame.
    Return: probability, label, feature contributions.
    """
    info   = trained_models[model_name]
    cols   = info["feature_cols"]
    scaler = info["scaler"]
    model  = info["model"]

    X   = pd.DataFrame([[float(customer_row.get(c, 0)) for c in cols]], columns=cols)
    X_s = scaler.transform(X)

    prob = float(model.predict_proba(X_s)[0][1])
    pred = prob >= 0.5

    # kontribusi tiap fitur
    if model_name == "Logistic Regression":
        contrib_raw = np.abs(model.coef_[0]) * np.abs(X_s[0])
    else:
        contrib_raw = model.feature_importances_ * np.abs(X_s[0])

    total = contrib_raw.sum() or 1
    contrib = {
        FEATURE_LABELS.get(c, c): round(float(v / total * 100), 1)
        for c, v in zip(cols, contrib_raw)
    }

    return {
        "probability":    round(prob, 4),
        "predicted_risk": pred,
        "risk_label":     "🔴 High Risk" if pred else "🟢 Low Risk",
        "contributions":  contrib,
    }


# ══════════════════════════════════════════════════════════
# PREDICT MANUAL INPUT (new customer)
# ══════════════════════════════════════════════════════════

def predict_manual(model_name: str, trained_models: dict, values: dict) -> dict:
    """Prediksi dari input manual (customer baru / tidak ada di DB)."""
    row = pd.Series(values)
    return predict_customer(model_name, trained_models, row)