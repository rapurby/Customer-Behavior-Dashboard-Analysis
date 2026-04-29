"""
data.py — All queries for DVD Rental Customer Behaviour Dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
from db import query


# ══════════════════════════════════════════════════════════════════════════
# CORE FACT TABLE
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_rentals() -> pd.DataFrame:
    sql = """
        SELECT
            r.rental_id,
            r.rental_date,
            r.return_date,
            EXTRACT(DOW FROM r.rental_date)        AS day_of_week_num,
            TO_CHAR(r.rental_date, 'Day')          AS day_of_week,
            CASE WHEN EXTRACT(DOW FROM r.rental_date) IN (0,6)
                 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            TO_CHAR(r.rental_date, 'YYYY-MM')      AS rental_month,
            c.customer_id,
            c.first_name || ' ' || c.last_name     AS full_name,
            c.email,
            c.active,
            ci.city,
            co.country,
            f.film_id,
            f.title,
            f.rental_duration                      AS allowed_days,
            f.rental_rate,
            f.length                               AS film_length_min,
            f.rating::text                         AS rating,
            cat.name                               AS category,
            p.amount,
            EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                                                   AS rental_duration_actual,
            GREATEST(
                EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                - f.rental_duration, 0
            )                                      AS days_overdue,
            CASE
                WHEN r.return_date IS NULL THEN NULL
                WHEN EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                     > f.rental_duration THEN TRUE
                ELSE FALSE
            END                                    AS is_late_return
        FROM rental r
        JOIN customer      c   ON r.customer_id   = c.customer_id
        JOIN address       a   ON c.address_id    = a.address_id
        JOIN city          ci  ON a.city_id       = ci.city_id
        JOIN country       co  ON ci.country_id   = co.country_id
        JOIN inventory     i   ON r.inventory_id  = i.inventory_id
        JOIN film          f   ON i.film_id       = f.film_id
        JOIN film_category fc  ON f.film_id       = fc.film_id
        JOIN category      cat ON fc.category_id  = cat.category_id
        LEFT JOIN payment  p   ON r.rental_id     = p.rental_id
        ORDER BY c.customer_id, r.rental_date
    """
    df = query(sql)
    df["rental_date"]    = pd.to_datetime(df["rental_date"])
    df["return_date"]    = pd.to_datetime(df["return_date"])
    df["is_late_return"] = df["is_late_return"].astype("boolean")
    df["days_overdue"]   = df["days_overdue"].fillna(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_films() -> pd.DataFrame:
    sql = """
        SELECT f.film_id, f.title, f.description, f.release_year,
               f.rental_duration, f.rental_rate, f.length,
               f.rating::text AS rating, f.replacement_cost,
               cat.name AS category
        FROM film f
        JOIN film_category fc  ON f.film_id      = fc.film_id
        JOIN category      cat ON fc.category_id = cat.category_id
        ORDER BY f.film_id
    """
    return query(sql)


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER 360 PROFILE
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_customer_profiles(rentals: pd.DataFrame) -> pd.DataFrame:
    """Build a full behavioural profile per customer."""
    today = rentals["rental_date"].max()

    # base aggregation
    base = rentals.groupby("customer_id").agg(
        full_name         =("full_name",      "first"),
        city              =("city",           "first"),
        country           =("country",        "first"),   # ← stored here
        active            =("active",         "first"),
        total_rentals     =("rental_id",      "count"),
        total_spending    =("amount",         "sum"),
        last_rental_date  =("rental_date",    "max"),
        first_rental_date =("rental_date",    "min"),
        late_return_count =("is_late_return", lambda x: x.fillna(False).sum()),
        avg_days_overdue  =("days_overdue",   "mean"),
        weekend_rentals   =("day_type",       lambda x: (x == "Weekend").sum()),
    ).reset_index()

    base["avg_spend_per_rental"]   = base["total_spending"] / base["total_rentals"]
    base["late_return_rate"]       = (base["late_return_count"] / base["total_rentals"]).round(4)
    base["days_since_last_rental"] = (today - base["last_rental_date"]).dt.days
    base["customer_tenure_days"]   = (today - base["first_rental_date"]).dt.days
    base["pct_weekend"]            = (base["weekend_rentals"] / base["total_rentals"] * 100).round(1)

    months_active = (
        (base["last_rental_date"].dt.to_period("M") -
         base["first_rental_date"].dt.to_period("M"))
        .apply(lambda x: max(x.n, 1))
    )
    base["avg_rentals_per_month"] = (base["total_rentals"] / months_active).round(2)

    # avg rental gap
    df_s = rentals.sort_values(["customer_id", "rental_date"])
    df_s["prev"] = df_s.groupby("customer_id")["rental_date"].shift(1)
    df_s["gap"]  = (df_s["rental_date"] - df_s["prev"]).dt.days
    gap_df = df_s.groupby("customer_id")["gap"].mean().reset_index()
    gap_df.columns = ["customer_id", "avg_rental_gap_days"]
    base = base.merge(gap_df, on="customer_id", how="left")

    # genre diversity + top genre
    genre_agg = rentals.groupby("customer_id").agg(
        unique_genres=("category", "nunique"),
        top_genre    =("category", lambda x: x.value_counts().index[0]),
    ).reset_index()
    totals = base.set_index("customer_id")["total_rentals"]
    genre_agg["genre_diversity"] = (
        genre_agg["unique_genres"] /
        genre_agg["customer_id"].map(totals)
    ).round(4)
    base = base.merge(genre_agg, on="customer_id", how="left")

    # risk score
    base["risk_score"] = (
        (base["late_return_rate"] * 0.5) +
        (base["avg_days_overdue"].clip(0, 10) / 10 * 0.3) +
        (base["avg_rental_gap_days"].fillna(30).clip(0, 60) / 60 * 0.2)
    )
    np.random.seed(42)
    base["risk_score"] = (
        base["risk_score"] + np.random.normal(0, 0.03, len(base))
    ).clip(0, 1)
    base["is_high_risk"] = base["risk_score"] > base["risk_score"].quantile(0.55)

    # spending tier
    q1, q2, q3 = base["total_spending"].quantile([0.25, 0.5, 0.75])
    def spend_tier(v):
        if v <= q1: return "Low Spender"
        if v <= q2: return "Mid Spender"
        if v <= q3: return "High Spender"
        return "Top Spender"
    base["spending_tier"] = base["total_spending"].apply(spend_tier)

    # RFM segment
    r_med = base["days_since_last_rental"].median()
    f_med = base["total_rentals"].median()
    m_med = base["total_spending"].median()
    def rfm_seg(row):
        n = sum([
            row["days_since_last_rental"] <= r_med,
            row["total_rentals"]          >= f_med,
            row["total_spending"]         >= m_med,
        ])
        return ["Lost", "At Risk", "Loyal", "Champions"][n]
    base["rfm_segment"] = base.apply(rfm_seg, axis=1)

    return base.fillna({"avg_rental_gap_days": 0, "genre_diversity": 0,
                        "unique_genres": 0, "top_genre": "Unknown"})


# ══════════════════════════════════════════════════════════════════════════
# PER-CUSTOMER HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_customer_rentals(cid: int, rentals: pd.DataFrame) -> pd.DataFrame:
    return rentals[rentals["customer_id"] == cid].copy()

def get_customer_timeline(cid: int, rentals: pd.DataFrame) -> pd.DataFrame:
    df = get_customer_rentals(cid, rentals).copy()
    return (
        df.groupby("rental_month")
        .agg(rentals =("rental_id",      "count"),
             spending=("amount",         "sum"),
             late    =("is_late_return", lambda x: x.fillna(False).sum()))
        .reset_index()
    )

def get_genre_preferences(cid: int, rentals: pd.DataFrame) -> pd.DataFrame:
    return (
        get_customer_rentals(cid, rentals)
        .groupby("category").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


# ══════════════════════════════════════════════════════════════════════════
# GENRE CO-OCCURRENCE (for predictions)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_genre_transition_matrix(rentals: pd.DataFrame) -> pd.DataFrame:
    """
    Build a category transition matrix: given a customer just rented genre A,
    what genre do they rent next? Used for next-category prediction.
    """
    df = rentals.sort_values(["customer_id", "rental_date"]).copy()
    df["next_category"] = df.groupby("customer_id")["category"].shift(-1)
    trans = (
        df.dropna(subset=["next_category"])
        .groupby(["category", "next_category"])
        .size().reset_index(name="count")
    )
    # normalize per row → probability
    trans["prob"] = trans.groupby("category")["count"].transform(lambda x: x / x.sum())
    return trans


# ══════════════════════════════════════════════════════════════════════════
# REGION / COUNTRY HELPERS  (fix: pull country from profiles, not rentals)
# ══════════════════════════════════════════════════════════════════════════

def get_country_genre_pivot(rentals: pd.DataFrame,
                            profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build country × genre share pivot.
    Merges country from profiles (safe) — not from rentals directly,
    which caused the KeyError when 'country' was missing after merge.
    """
    # rentals already has country from the SQL, but we re-attach from
    # profiles to guarantee the column exists post-merge operations
    cntry_map = profiles.set_index("customer_id")["country"]
    df = rentals.copy()
    df["country"] = df["customer_id"].map(cntry_map)
    df = df.dropna(subset=["country"])

    ctry_genre = (
        df.groupby(["country", "category"])
        .size().reset_index(name="rentals")
    )
    ctry_genre["share"] = (
        ctry_genre.groupby("country")["rentals"]
        .transform(lambda x: x / x.sum() * 100)
        .round(1)
    )
    pivot = ctry_genre.pivot_table(
        index="country", columns="category",
        values="share", fill_value=0,
    )
    pivot.columns.name = None
    return pivot


def get_country_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles.groupby("country")
        .agg(
            customers   =("customer_id",     "count"),
            avg_spending=("total_spending",   "mean"),
            avg_late    =("late_return_rate", "mean"),
            high_risk   =("is_high_risk",     "sum"),
        )
        .reset_index()
        .sort_values("customers", ascending=False)
    )


# ══════════════════════════════════════════════════════════════════════════
# BUSINESS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_monthly_trend() -> pd.DataFrame:
    sql = """
        SELECT
            TO_CHAR(r.rental_date, 'YYYY-MM')   AS month,
            COUNT(r.rental_id)                  AS rentals,
            COUNT(DISTINCT r.customer_id)       AS unique_customers,
            SUM(p.amount)                       AS revenue,
            SUM(CASE WHEN r.return_date IS NOT NULL AND
                     EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                     > f.rental_duration THEN 1 ELSE 0 END) AS late_returns
        FROM rental r
        JOIN inventory i ON r.inventory_id = i.inventory_id
        JOIN film      f ON i.film_id      = f.film_id
        LEFT JOIN payment p ON r.rental_id = p.rental_id
        GROUP BY month ORDER BY month
    """
    return query(sql)


@st.cache_data(ttl=300, show_spinner=False)
def get_category_stats() -> pd.DataFrame:
    sql = """
        SELECT
            cat.name                          AS category,
            COUNT(r.rental_id)                AS rentals,
            COUNT(DISTINCT r.customer_id)     AS unique_customers,
            SUM(p.amount)                     AS revenue,
            AVG(p.amount)                     AS avg_payment,
            SUM(CASE WHEN r.return_date IS NOT NULL AND
                     EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                     > f.rental_duration THEN 1 ELSE 0 END) AS late_count,
            AVG(GREATEST(
                EXTRACT(EPOCH FROM (r.return_date - r.rental_date))/86400.0
                - f.rental_duration, 0))      AS avg_overdue_days
        FROM rental r
        JOIN inventory     i   ON r.inventory_id  = i.inventory_id
        JOIN film          f   ON i.film_id       = f.film_id
        JOIN film_category fc  ON f.film_id       = fc.film_id
        JOIN category      cat ON fc.category_id  = cat.category_id
        LEFT JOIN payment  p   ON r.rental_id     = p.rental_id
        GROUP BY cat.name ORDER BY rentals DESC
    """
    return query(sql)


@st.cache_data(ttl=300, show_spinner=False)
def get_dayofweek_stats() -> pd.DataFrame:
    sql = """
        SELECT
            EXTRACT(DOW FROM r.rental_date)   AS dow_num,
            TO_CHAR(r.rental_date, 'Dy')      AS day_label,
            CASE WHEN EXTRACT(DOW FROM r.rental_date) IN (0,6)
                 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            COUNT(r.rental_id)                AS rentals,
            SUM(p.amount)                     AS revenue
        FROM rental r
        LEFT JOIN payment p ON r.rental_id = p.rental_id
        GROUP BY dow_num, day_label, day_type
        ORDER BY dow_num
    """
    return query(sql)