"""
app.py — DVD Rental Customer Behaviour Dashboard
Run:  streamlit run app.py

NARRATIVE FLOW:
  1. Business Snapshot       → "How is the business doing?"
  2. Customers & Segments    → "Who rents? Full table + 360 profile merged."
  3. Growth Opportunities    → "Where to act to grow revenue."
  4. Predictions & Risk      → "What will happen? Who needs attention?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import test_connection, DB_CONFIG
from data import (
    get_rentals, get_films,
    get_customer_profiles,
    get_customer_rentals, get_customer_timeline, get_genre_preferences,
    get_monthly_trend, get_category_stats, get_dayofweek_stats,
    get_country_genre_pivot, get_country_summary,
    get_genre_transition_matrix,
)
from models import train_all_models, predict_manual, FEATURE_SETS, FEATURE_LABELS

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DVD Rental · Customer Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — professional, clean
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding: 1.4rem 2.5rem 2.5rem; }

  /* ── Sidebar: dark, minimal, professional ── */
  [data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div { color: #d1d5db !important; }
  [data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
  [data-testid="stSidebar"] .stRadio label {
    font-size: 0.84rem !important;
    padding: 7px 10px !important;
    border-radius: 6px !important;
    transition: background 0.15s;
  }
  [data-testid="stSidebar"] .stRadio label:hover {
    background: #1f2937 !important;
  }

  /* ── Metrics ── */
  [data-testid="metric-container"] {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  [data-testid="metric-container"] label {
    color: #6b7280 !important; font-size: 0.75rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: .04em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important; font-weight: 700; font-size: 1.35rem;
  }

  /* ── Page header ── */
  .ph { border-left: 4px solid #2563eb; padding: 4px 0 4px 14px; margin-bottom: 10px; }
  .ph-title { font-size: 1.4rem; font-weight: 700; color: #111827; margin: 0; }
  .ph-sub   { font-size: 0.83rem; color: #6b7280; margin: 3px 0 0; }

  /* ── Section label ── */
  .sl { font-size: 0.69rem; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #2563eb; margin: 18px 0 5px; display: block; }

  /* ── Chart title ── */
  .ct  { font-size: 0.88rem; font-weight: 600; color: #1f2937; margin-bottom: 1px; }
  .cst { font-size: 0.74rem; color: #9ca3af; margin-bottom: 7px; }

  /* ── Callouts ── */
  .co-blue  { background:#eff6ff; border-left:3px solid #3b82f6; border-radius:0 7px 7px 0;
    padding:10px 14px; font-size:0.82rem; color:#1e40af; margin:8px 0; }
  .co-amber { background:#fffbeb; border-left:3px solid #f59e0b; border-radius:0 7px 7px 0;
    padding:10px 14px; font-size:0.82rem; color:#92400e; margin:8px 0; }
  .co-green { background:#f0fdf4; border-left:3px solid #22c55e; border-radius:0 7px 7px 0;
    padding:10px 14px; font-size:0.82rem; color:#166534; margin:8px 0; }
  .co-red   { background:#fef2f2; border-left:3px solid #ef4444; border-radius:0 7px 7px 0;
    padding:10px 14px; font-size:0.82rem; color:#7f1d1d; margin:8px 0; }

  /* ── Badges ── */
  .pill  { display:inline-block; background:#f3f4f6; border-radius:20px;
           padding:2px 10px; font-size:0.76rem; color:#374151; margin:2px 3px 2px 0; }
  .b-hi  { background:#fee2e2; color:#b91c1c; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .b-lo  { background:#dcfce7; color:#15803d; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .s-ch  { background:#fef9c3; color:#713f12; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .s-lo  { background:#dbeafe; color:#1e3a8a; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .s-ar  { background:#fee2e2; color:#991b1b; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .s-ls  { background:#f3f4f6; color:#4b5563; border-radius:20px;
           padding:2px 10px; font-size:0.75rem; font-weight:600; }

  /* ── Read-only field ── */
  .ro     { background:#f9fafb; border:1px solid #e5e7eb; border-radius:7px;
            padding:9px 12px; margin-bottom:7px; }
  .ro-lbl { font-size:0.71rem; color:#6b7280; font-weight:500; margin-bottom:1px; }
  .ro-val { font-size:0.94rem; font-weight:700; color:#111827; }

  /* ── Action card ── */
  .ac { background:#fff; border:1px solid #e5e7eb; border-radius:9px;
        padding:12px 14px; margin-bottom:8px; }
  .ac-t { font-size:0.86rem; font-weight:700; color:#111827; margin-bottom:3px; }
  .ac-b { font-size:0.8rem; color:#4b5563; }

  /* ── Prediction card ── */
  .pc { background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;
        padding:13px 15px; margin-bottom:9px; }

  /* ── Insight cards (3-tier) ── */
  .ins-imm  { background:#fef2f2; border:1px solid #fca5a5; border-radius:9px;
              padding:13px 15px; margin-bottom:8px; }
  .ins-short{ background:#fffbeb; border:1px solid #fde68a; border-radius:9px;
              padding:13px 15px; margin-bottom:8px; }
  .ins-long { background:#f0fdf4; border:1px solid #86efac; border-radius:9px;
              padding:13px 15px; margin-bottom:8px; }
  .ins-t    { font-size:0.84rem; font-weight:700; margin-bottom:3px; }
  .ins-b    { font-size:0.79rem; color:#374151; }

  /* ── Segment progress bar ── */
  .seg-bar  { height:8px; border-radius:99px; margin:4px 0; }

  #MainMenu { visibility:hidden; }
  footer    { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

C = dict(blue="#2563eb", indigo="#4338ca", green="#16a34a",
         red="#dc2626", amber="#d97706", slate="#64748b", sky="#0284c7")
SEG_COLORS = {"Champions": "#eab308", "Loyal": "#3b82f6",
              "At Risk": "#ef4444", "Lost": "#94a3b8"}


# ══════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════
def ph(title, sub=""):
    st.markdown(
        f'<div class="ph"><div class="ph-title">{title}</div>'
        f'<div class="ph-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def sl(text):
    st.markdown(f'<span class="sl">{text}</span>', unsafe_allow_html=True)

def cc(title, sub=""):
    st.markdown(f'<div class="ct">{title}</div><div class="cst">{sub}</div>',
                unsafe_allow_html=True)

def co(text, kind="blue"):
    st.markdown(f'<div class="co-{kind}">{text}</div>', unsafe_allow_html=True)

def pill(t):   return f'<span class="pill">{t}</span>'
def rb(hi):    return '<span class="b-hi">⚠ High Risk</span>' if hi else '<span class="b-lo">✓ Low Risk</span>'
def sb(seg):
    m = {"Champions": "ch", "Loyal": "lo", "At Risk": "ar", "Lost": "ls"}
    return f'<span class="s-{m.get(seg,"ls")}">{seg}</span>'

def ro(lbl, val):
    st.markdown(f'<div class="ro"><div class="ro-lbl">{lbl}</div>'
                f'<div class="ro-val">{val}</div></div>', unsafe_allow_html=True)

def ac(icon, title, body):
    st.markdown(f'<div class="ac"><div class="ac-t">{icon} {title}</div>'
                f'<div class="ac-b">{body}</div></div>', unsafe_allow_html=True)

def insight_card(kind, label, title, body):
    # kind: imm | short | long
    st.markdown(
        f'<div class="ins-{kind}"><div class="ins-t">{label} {title}</div>'
        f'<div class="ins-b">{body}</div></div>',
        unsafe_allow_html=True,
    )

def cf(fig, h=300):
    fig.update_layout(
        height=h, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#374151"),
        margin=dict(l=8, r=8, t=34, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6",
                   linecolor="#e5e7eb", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6",
                   linecolor="#e5e7eb", zeroline=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# CONNECTION + SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
conn_ok = test_connection()

PAGES = [
    ("📊", "Business Snapshot"),
    ("👥", "Customers & Segments"),
    ("💡", "Growth Opportunities"),
    ("🔮", "Predictions & Risk"),
]
page_labels = [f"{i} {n}" for i, n in PAGES]

with st.sidebar:
    # Logo / brand
    st.markdown(
        '<div style="padding:18px 16px 12px;border-bottom:1px solid #1f2937">'
        '<div style="font-size:1.05rem;font-weight:700;color:#f9fafb;'
        'letter-spacing:.02em">🎬 DVD Rental</div>'
        '<div style="font-size:0.75rem;color:#6b7280;margin-top:2px">'
        'Customer Analytics</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if conn_ok is True:
        st.markdown(
            '<div style="background:#052e16;color:#86efac;border-radius:5px;'
            'padding:5px 10px;font-size:0.76rem;margin:0 0 12px">● Database Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#450a0a;color:#fca5a5;border-radius:5px;'
            'padding:5px 10px;font-size:0.76rem;margin:0 0 12px">● Disconnected — edit db.py</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Config"):
            st.code(f"host:     {DB_CONFIG['host']}\nport:     {DB_CONFIG['port']}\n"
                    f"database: {DB_CONFIG['database']}\nuser:     {DB_CONFIG['user']}",
                    language="yaml")
        st.stop()

    st.markdown(
        '<div style="font-size:0.68rem;font-weight:600;letter-spacing:.1em;'
        'text-transform:uppercase;color:#4b5563;padding:0 4px 5px">Navigation</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("", page_labels, label_visibility="collapsed")
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
with st.spinner("Loading data from database…"):
    rentals  = get_rentals()
    films    = get_films()
    profiles = get_customer_profiles(rentals)
    trans_mx = get_genre_transition_matrix(rentals)

with st.sidebar:
    st.markdown(
        f'<div style="border-top:1px solid #1f2937;padding:12px 4px 0;'
        f'font-size:0.73rem;color:#4b5563">'
        f'👥 {len(profiles):,} customers<br>'
        f'🎬 {len(films):,} films<br>'
        f'🎟️ {len(rentals):,} rentals</div>',
        unsafe_allow_html=True,
    )

with st.spinner("Training ML models…"):
    trained_models = train_all_models(profiles)

ALL_CATEGORIES = sorted(rentals["category"].unique())
ALL_COUNTRIES  = sorted(profiles["country"].dropna().unique())


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — BUSINESS SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════
if page == page_labels[0]:
    ph("📊 Business Snapshot",
       "Top-level KPIs and trends — start here to understand overall business health.")

    total_cust = len(profiles)
    total_rev  = float(rentals["amount"].sum())
    total_rent = len(rentals)
    avg_rev_c  = total_rev / total_cust
    active_c   = int(profiles["active"].sum())
    # Late return rate: rentals returned after allowed_days
    late_n     = int(rentals["is_late_return"].fillna(False).sum())
    returned   = int(rentals["return_date"].notna().sum())
    late_pct   = late_n / returned * 100 if returned else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue",          f"${total_rev:,.0f}")
    k2.metric("Total Rentals",          f"{total_rent:,}")
    k3.metric("Total Customers",        f"{total_cust:,}")
    k4.metric("Avg Revenue / Customer", f"${avg_rev_c:.2f}")
    k5.metric("Late Return Rate",       f"{late_pct:.1f}%",
              delta_color="inverse",
              help=f"Rentals returned after the allowed rental period ÷ total returned rentals. "
                   f"= {late_n:,} late ÷ {returned:,} returned.")

    st.markdown("---")

    monthly = get_monthly_trend()
    sl("RENTAL VOLUME & REVENUE TREND")
    cc("Monthly Rental Activity", "Rentals, revenue, and unique customers over time.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["rentals"],
                             name="Rentals", mode="lines+markers",
                             line=dict(color=C["blue"], width=2.5),
                             marker=dict(size=5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["revenue"],
                             name="Revenue ($)", mode="lines+markers",
                             line=dict(color=C["green"], width=2.5, dash="dot"),
                             marker=dict(size=5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["unique_customers"],
                             name="Active Customers", mode="lines+markers",
                             line=dict(color=C["amber"], width=1.5, dash="dashdot"),
                             marker=dict(size=4)), secondary_y=False)
    fig.update_layout(height=290, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter", size=12),
                      margin=dict(l=8, r=8, t=10, b=8),
                      legend=dict(orientation="h", y=1.06),
                      xaxis=dict(showgrid=True, gridcolor="#f3f4f6"))
    fig.update_yaxes(title_text="Rentals / Customers", secondary_y=False,
                     showgrid=True, gridcolor="#f3f4f6")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    peak_m = monthly.loc[monthly["rentals"].idxmax(), "month"]
    co(f"📈 Peak rental month: <b>{peak_m}</b>. Plan stock and campaigns around this period.",
       "blue")

    st.markdown("---")
    l, r = st.columns(2)
    with l:
        dow = get_dayofweek_stats()
        sl("RENTALS BY DAY OF WEEK")
        cc("When Do Customers Rent?", "Weekday vs. weekend volume.")
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow["day_label"] = dow["day_label"].str.strip()
        dow_s = dow.set_index("day_label").reindex(day_order).reset_index()
        fig = px.bar(dow_s, x="day_label", y="rentals",
                     color="day_type",
                     color_discrete_map={"Weekday": C["blue"], "Weekend": C["amber"]},
                     labels={"day_label": "", "rentals": "Rentals", "day_type": ""},
                     text="rentals")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        cf(fig, 260)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        cat_df = get_category_stats()
        sl("REVENUE BY FILM CATEGORY")
        cc("Which Genres Drive Revenue?", "Total revenue per category.")
        cat_rev = cat_df.sort_values("revenue", ascending=True)
        fig = px.bar(cat_rev, x="revenue", y="category", orientation="h",
                     color="revenue",
                     color_continuous_scale=["#dbeafe", C["indigo"]],
                     labels={"revenue": "Total Revenue ($)", "category": ""},
                     text=cat_rev["revenue"].apply(lambda v: f"${v:,.0f}"))
        fig.update_traces(textposition="outside")
        cf(fig, 260)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    co("💡 <b>Next →</b> Go to <b>Customers & Segments</b> to explore the people behind these numbers.",
       "blue")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMERS & SEGMENTS  (merged page 2 + 3)
# ══════════════════════════════════════════════════════════════════════════
elif page == page_labels[1]:
    ph("👥 Customers & Segments",
       "Filter and explore the customer base, then open any customer's full 360° profile.")

    # ── FILTERS ──────────────────────────────────────────────────────────
    sl("FILTER CUSTOMERS")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        filt_seg = st.multiselect("RFM Segment",
            ["Champions", "Loyal", "At Risk", "Lost"], default=[],
            placeholder="All segments")
    with f2:
        filt_risk = st.selectbox("Risk Level",
            ["All", "High Risk Only", "Low Risk Only"])
    with f3:
        filt_region = st.multiselect("Country",
            ALL_COUNTRIES, default=[], placeholder="All countries")
    with f4:
        filt_tier = st.multiselect("Spending Tier",
            ["Low Spender", "Mid Spender", "High Spender", "Top Spender"],
            default=[], placeholder="All tiers")
    with f5:
        sort_by = st.selectbox("Sort by",
            ["Name (A–Z)", "Name (Z–A)", "Spending ↓", "Spending ↑",
             "Rentals ↓", "Rentals ↑", "Risk Score ↓"])

    display = profiles.copy()
    if filt_seg:    display = display[display["rfm_segment"].isin(filt_seg)]
    if filt_risk == "High Risk Only": display = display[display["is_high_risk"] == True]
    elif filt_risk == "Low Risk Only":display = display[display["is_high_risk"] == False]
    if filt_region: display = display[display["country"].isin(filt_region)]
    if filt_tier:   display = display[display["spending_tier"].isin(filt_tier)]

    sort_map = {
        "Name (A–Z)":   ("full_name",      True),
        "Name (Z–A)":   ("full_name",      False),
        "Spending ↓":   ("total_spending", False),
        "Spending ↑":   ("total_spending", True),
        "Rentals ↓":    ("total_rentals",  False),
        "Rentals ↑":    ("total_rentals",  True),
        "Risk Score ↓": ("risk_score",     False),
    }
    sc, asc = sort_map[sort_by]
    display = display.sort_values(sc, ascending=asc).reset_index(drop=True)

    # ── SEGMENT CARDS ─────────────────────────────────────────────────────
    st.markdown("---")
    sl("SEGMENT OVERVIEW")
    seg_defs = {
        "Champions": ("🏆", C["amber"],   "High recency, frequency & spend"),
        "Loyal":     ("💙", C["blue"],    "Regular renters — high potential"),
        "At Risk":   ("⚠️", C["red"],     "Declining activity — act now"),
        "Lost":      ("💤", "#94a3b8",    "Low across all dimensions"),
    }
    sc4 = st.columns(4)
    for col, (seg, (icon, color, desc)) in zip(sc4, seg_defs.items()):
        cnt = int((profiles["rfm_segment"] == seg).sum())
        rev = float(profiles[profiles["rfm_segment"] == seg]["total_spending"].sum())
        pct = cnt / len(profiles) * 100
        col.markdown(
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;'
            f'padding:14px 15px">'
            f'<div style="font-size:1.1rem">{icon}</div>'
            f'<div style="font-weight:700;font-size:0.93rem;color:#111827;margin-top:4px">{seg}</div>'
            f'<div style="font-size:1.55rem;font-weight:800;color:{color};line-height:1.2">{cnt:,}</div>'
            f'<div style="background:#f3f4f6;border-radius:99px;height:5px;margin:6px 0">'
            f'<div style="background:{color};width:{pct:.0f}%;height:5px;border-radius:99px"></div></div>'
            f'<div style="font-size:0.72rem;color:#6b7280">{pct:.1f}% · ${rev:,.0f} revenue</div>'
            f'<div style="font-size:0.72rem;color:#9ca3af;margin-top:2px">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── CHARTS ────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        sl("FREQUENCY VS. SPENDING")
        cc("Rental Frequency vs. Total Spending",
           "Each dot = one customer. Color = RFM segment.")
        fig = px.scatter(display, x="total_rentals", y="total_spending",
                         color="rfm_segment", color_discrete_map=SEG_COLORS,
                         opacity=0.65,
                         labels={"total_rentals": "Total Rentals",
                                 "total_spending": "Total Spending ($)",
                                 "rfm_segment": "Segment"},
                         hover_data={"full_name": True, "rfm_segment": True,
                                     "total_rentals": True, "total_spending": True})
        cf(fig, 270)
        fig.update_traces(marker=dict(size=7, opacity=0.65,
                                      line=dict(width=0.5, color="white")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        sl("SPENDING TIER BREAKDOWN")
        cc("Customers by Spending Tier", "Distribution of (filtered) customers by value.")
        tier_order  = ["Low Spender", "Mid Spender", "High Spender", "Top Spender"]
        tier_colors = {"Low Spender": "#9ca3af", "Mid Spender": "#60a5fa",
                       "High Spender": "#3b82f6", "Top Spender": "#1d4ed8"}
        tc = display["spending_tier"].value_counts().reindex(tier_order, fill_value=0)
        fig = go.Figure(go.Bar(
            x=tc.index, y=tc.values,
            marker_color=[tier_colors[t] for t in tc.index],
            text=tc.values, textposition="outside",
        ))
        cf(fig, 270)
        fig.update_layout(showlegend=False, yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── CUSTOMER TABLE + PROFILE DETAIL ───────────────────────────────────
    sl(f"CUSTOMER TABLE — {len(display):,} RESULTS")
    co("Click <b>Profile</b> on any row to view the full 360° profile for that customer.",
       "blue")

    if "cust_id" not in st.session_state:
        st.session_state.cust_id = None

    # TABLE VIEW
    if st.session_state.cust_id is None:
        if display.empty:
            st.info("No customers match the current filters.")
        else:
            hdr = st.columns([0.45, 2, 1.3, 0.75, 1, 0.85, 1, 1, 0.9])
            for h, lbl in zip(hdr, ["ID","Name","Location","Rentals",
                                     "Spending","Late %","Segment","Risk","Action"]):
                h.markdown(f"<b style='font-size:0.75rem'>{lbl}</b>",
                           unsafe_allow_html=True)
            st.divider()
            for _, r in display.head(200).iterrows():
                row = st.columns([0.45, 2, 1.3, 0.75, 1, 0.85, 1, 1, 0.9])
                row[0].markdown(f"#{int(r.customer_id)}")
                row[1].markdown(r.full_name)
                row[2].markdown(f"{r.city}, {r.country}")
                row[3].markdown(f"{int(r.total_rentals)}")
                row[4].markdown(f"${float(r.total_spending):.0f}")
                row[5].markdown(f"{float(r.late_return_rate)*100:.0f}%")
                row[6].markdown(sb(r.rfm_segment), unsafe_allow_html=True)
                row[7].markdown(rb(bool(r.is_high_risk)), unsafe_allow_html=True)
                if row[8].button("Profile", key=f"t_{r.customer_id}"):
                    st.session_state.cust_id = int(r.customer_id)
                    st.rerun()
            if len(display) > 200:
                st.caption(f"Showing first 200 of {len(display):,}. Refine filters.")

    # PROFILE DETAIL VIEW
    else:
        if st.button("← Back to customer list"):
            st.session_state.cust_id = None
            st.rerun()

        cid  = st.session_state.cust_id
        crow = profiles[profiles["customer_id"] == cid]
        if crow.empty:
            st.error(f"Customer #{cid} not found.")
            st.session_state.cust_id = None
            st.stop()
        c      = crow.iloc[0]
        c_rent = get_customer_rentals(cid, rentals)

        st.markdown("---")
        # identity strip
        av_color = C["red"] if c.is_high_risk else C["green"]
        name_parts = str(c["full_name"]).split()
        initials = "".join([p[0] for p in name_parts[:2]]).upper()
        ic1, ic2 = st.columns([0.33, 3.67])
        with ic1:
            st.markdown(
                f'<div style="background:{av_color};color:white;border-radius:50%;'
                f'width:66px;height:66px;display:flex;align-items:center;'
                f'justify-content:center;font-size:1.45rem;font-weight:700">{initials}</div>',
                unsafe_allow_html=True,
            )
        with ic2:
            st.markdown(f"## {c.full_name}")
            st.markdown(
                pill(f"ID #{cid}") + pill(f"{c.city}, {c.country}") +
                sb(c.rfm_segment) + " &nbsp; " + rb(bool(c.is_high_risk)),
                unsafe_allow_html=True,
            )

        st.markdown("---")
        sl("CUSTOMER HEALTH METRICS")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Rentals",    f"{int(c.total_rentals)}",
                  f"{float(c.avg_rentals_per_month):.1f}/month")
        m2.metric("Total Spending",   f"${float(c.total_spending):.2f}")
        m3.metric("Avg per Rental",   f"${float(c.avg_spend_per_rental):.2f}")
        m4.metric("Customer Since",
                  pd.to_datetime(c.first_rental_date).strftime("%b %Y"))
        m5.metric("Weekend Rentals",  f"{float(c.pct_weekend):.0f}%")

        st.markdown("---")
        sl("RENTAL ACTIVITY TIMELINE")
        cc("Monthly Rentals, Spending & Late Returns",
           "Rising bars = growing engagement. Red = late returns.")
        timeline = get_customer_timeline(cid, rentals)
        if not timeline.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=timeline["rental_month"], y=timeline["rentals"],
                                 name="Rentals", marker_color=C["blue"], opacity=0.75),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=timeline["rental_month"], y=timeline["spending"],
                                     name="Spending ($)", mode="lines+markers",
                                     line=dict(color=C["green"], width=2)),
                          secondary_y=True)
            fig.add_trace(go.Bar(x=timeline["rental_month"],
                                 y=timeline["late"].astype(float),
                                 name="Late Returns", marker_color=C["red"], opacity=0.55),
                          secondary_y=False)
            fig.update_layout(height=250, plot_bgcolor="white", paper_bgcolor="white",
                              barmode="overlay", font=dict(family="Inter", size=12),
                              margin=dict(l=8, r=8, t=10, b=8),
                              legend=dict(orientation="h", y=1.06),
                              xaxis=dict(showgrid=True, gridcolor="#f3f4f6"))
            fig.update_yaxes(title_text="Rentals", secondary_y=False)
            fig.update_yaxes(title_text="Spending ($)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        d1, d2, d3 = st.columns(3)
        with d1:
            sl("GENRE PREFERENCES")
            cc("What Does This Customer Watch?")
            genres = get_genre_preferences(cid, rentals)
            if not genres.empty:
                fig = px.pie(genres, names="category", values="count", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(height=230, plot_bgcolor="white", paper_bgcolor="white",
                                  margin=dict(l=0, r=0, t=8, b=0),
                                  legend=dict(font=dict(size=9)), font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)
                co(f"Top genre: <b>{genres.iloc[0]['category']}</b>. "
                   f"Unique genres: <b>{int(c.unique_genres)}</b>.", "green")

        with d2:
            sl("SPENDING BY GENRE")
            cc("Where Does the Money Go?")
            sg = (c_rent.groupby("category")["amount"].sum()
                  .reset_index().sort_values("amount", ascending=True))
            fig = px.bar(sg, x="amount", y="category", orientation="h",
                         color="amount",
                         color_continuous_scale=["#dbeafe", C["indigo"]],
                         labels={"amount": "Total Spent ($)", "category": ""},
                         text=sg["amount"].apply(lambda v: f"${v:.0f}"))
            fig.update_traces(textposition="outside")
            cf(fig, 230)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with d3:
            sl("RETURN BEHAVIOUR")
            cc("On-Time vs. Late Returns")
            late_c = int(c.late_return_count)
            ok_c   = int(c.total_rentals) - late_c
            fig = go.Figure(go.Pie(
                labels=["On Time", "Late"], values=[ok_c, late_c],
                hole=0.55, marker_colors=[C["green"], C["red"]], textinfo="label+percent"))
            fig.update_layout(height=230, showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                              font=dict(family="Inter", size=12),
                              annotations=[dict(text=f"<b>{int(c.total_rentals)}</b><br>Rentals",
                                                x=0.5, y=0.5, font_size=13, showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
            if late_c > 0:
                co(f"Late {late_c}× · avg {float(c.avg_days_overdue):.1f}d overdue.",
                   "amber" if not c.is_high_risk else "red")
            else:
                co("Perfect record — never late. 🎉", "green")

        # ── COMPANY ACTION RECOMMENDATIONS ──────────────────────────────────
        st.markdown("---")
        sl("RECOMMENDED COMPANY ACTIONS FOR THIS CUSTOMER")

        # Build action logic based on customer profile
        late_r = float(c.late_return_rate)
        days_away = int(c.days_since_last_rental)
        seg = c.rfm_segment
        hi_risk = bool(c.is_high_risk)
        spend_t = c.spending_tier
        total_r = int(c.total_rentals)

        # Determine actions dynamically
        actions_imm, actions_short, actions_long = [], [], []

        # --- IMMEDIATE ---
        if hi_risk and late_r > 0.3:
            actions_imm.append(("📱 Send Return Reminder",
                f"This customer has a {late_r*100:.0f}% late return rate. "
                "Set up an automated SMS/email reminder 1 day before each due date immediately."))
        if days_away > 60 and seg in ["At Risk", "Lost"]:
            actions_imm.append(("📧 Re-activation Email",
                f"Last rental was {days_away} days ago — this customer is drifting. "
                "Send a personalised 'We miss you' email with a 15% discount voucher today."))
        if seg == "Lost" and days_away > 120:
            actions_imm.append(("🎁 Win-Back Offer",
                f"Customer has been inactive for {days_away} days. "
                "A bold offer (e.g. 1 free rental) is needed to re-activate before permanent churn."))
        if not actions_imm:
            actions_imm.append(("✅ No Urgent Action Needed",
                "This customer is currently engaged. Maintain relationship with regular communication."))

        # --- SHORT-TERM ---
        if seg == "Champions":
            actions_short.append(("🏆 Enrol in VIP Loyalty Programme",
                f"This customer has rented {total_r} times and is a top spender ({spend_t}). "
                "Invite them to an exclusive VIP tier with early access to new releases."))
        if hi_risk:
            actions_short.append(("📋 Limit Concurrent Rentals",
                "Cap at 2 active rentals until on-time return rate improves. "
                "Review monthly — restore full access when late rate drops below 20%."))
        if seg == "Loyal" and not hi_risk:
            actions_short.append(("⬆️ Upsell to Premium Plan",
                f"Loyal customer with {spend_t} spending. Offer a subscription plan "
                "or 'rent 5 get 1 free' card to increase visit frequency."))
        if seg in ["At Risk"] and not actions_short:
            actions_short.append(("🎯 Genre-Targeted Promotion",
                f"Customer prefers {c.top_genre}. Send a curated list of new arrivals "
                "in their favourite genre with a limited-time discount to re-engage."))
        if not actions_short:
            actions_short.append(("📊 Monitor Engagement",
                "Track rental frequency monthly. If gap between rentals widens by 50%, "
                "escalate to an At-Risk re-engagement campaign."))

        # --- LONG-TERM ---
        if spend_t in ["Top Spender", "High Spender"]:
            actions_long.append(("💰 Maximise Lifetime Value",
                f"At {spend_t} level, this customer is high-value. "
                "Build a long-term relationship through personalised outreach, "
                "birthday offers, and exclusive catalogue previews."))
        if float(c.genre_diversity) < 0.2:
            actions_long.append(("🎭 Broaden Genre Exposure",
                f"Genre diversity score: {float(c.genre_diversity):.2f} — customer rents from "
                "a narrow set of genres. Gradual cross-genre recommendations can increase "
                "basket size and long-term retention."))
        if seg in ["Lost", "At Risk"]:
            actions_long.append(("🔄 Churn Prevention System",
                "Flag this customer in the automated churn monitoring system. "
                "If no rental in the next 30 days, trigger escalated intervention workflow."))
        if not actions_long:
            actions_long.append(("🌱 Develop Brand Loyalty",
                "Consistently engaged customers become brand advocates. "
                "A referral programme ('refer a friend, both get a free rental') "
                "can turn this customer into a growth channel."))

        ia, sa, la = st.columns(3)
        with ia:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;color:#dc2626;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">'
                '🚨 Immediate Action</div>',
                unsafe_allow_html=True,
            )
            for title, body in actions_imm:
                st.markdown(
                    f'<div style="background:#fef2f2;border:1px solid #fca5a5;'
                    f'border-radius:8px;padding:11px 13px;margin-bottom:7px">'
                    f'<div style="font-weight:700;font-size:0.83rem;color:#991b1b;'
                    f'margin-bottom:3px">{title}</div>'
                    f'<div style="font-size:0.78rem;color:#374151">{body}</div></div>',
                    unsafe_allow_html=True,
                )
        with sa:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;color:#d97706;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">'
                '📅 Short-Term (1–4 weeks)</div>',
                unsafe_allow_html=True,
            )
            for title, body in actions_short:
                st.markdown(
                    f'<div style="background:#fffbeb;border:1px solid #fde68a;'
                    f'border-radius:8px;padding:11px 13px;margin-bottom:7px">'
                    f'<div style="font-weight:700;font-size:0.83rem;color:#92400e;'
                    f'margin-bottom:3px">{title}</div>'
                    f'<div style="font-size:0.78rem;color:#374151">{body}</div></div>',
                    unsafe_allow_html=True,
                )
        with la:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;color:#16a34a;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">'
                '📈 Long-Term (1–3 months)</div>',
                unsafe_allow_html=True,
            )
            for title, body in actions_long:
                st.markdown(
                    f'<div style="background:#f0fdf4;border:1px solid #86efac;'
                    f'border-radius:8px;padding:11px 13px;margin-bottom:7px">'
                    f'<div style="font-weight:700;font-size:0.83rem;color:#166534;'
                    f'margin-bottom:3px">{title}</div>'
                    f'<div style="font-size:0.78rem;color:#374151">{body}</div></div>',
                    unsafe_allow_html=True,
                )

        # film recs
        st.markdown("---")
        sl("PERSONALISED FILM RECOMMENDATIONS")
        cc("Films Not Yet Rented", "Based on genre preferences, excluding already-rented films.")
        genres_pref = get_genre_preferences(cid, rentals)
        rented_ids  = set(c_rent["film_id"].tolist())
        cands = films[~films["film_id"].isin(rented_ids)].copy()
        genre_rank = {r["category"]: i for i, r in genres_pref.iterrows()}
        cands["genre_rank"] = cands["category"].map(genre_rank).fillna(999)
        if bool(c.is_high_risk):
            cands = cands.sort_values(["genre_rank", "rental_duration", "rental_rate"])
        else:
            cands = cands.sort_values(["genre_rank", "rental_rate"], ascending=[True, False])
        recs = (cands.groupby("category", group_keys=False).head(2)
                .sort_values(["genre_rank", "category"]).head(12))

        rh = st.columns([2.5, 1.2, 0.7, 1, 1, 1.6])
        for h, lbl in zip(rh, ["Film", "Genre", "Rating", "Period", "Price", "Note"]):
            h.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)
        st.divider()
        last_cat = None
        for _, fr in recs.iterrows():
            if fr["category"] != last_cat:
                st.markdown(f"<div style='font-size:0.76rem;font-weight:600;"
                            f"color:{C['blue']};margin:5px 0 2px'>🎭 {fr['category']}</div>",
                            unsafe_allow_html=True)
                last_cat = fr["category"]
            rc = st.columns([2.5, 1.2, 0.7, 1, 1, 1.6])
            rc[0].markdown(f"**{fr['title']}**")
            rc[1].markdown(fr["category"])
            rc[2].markdown(fr["rating"])
            rc[3].markdown(f"{int(fr['rental_duration'])} days")
            rc[4].markdown(f"${fr['rental_rate']:.2f}")
            note = ("✅ Short-term" if bool(c.is_high_risk) and fr["rental_duration"] <= 4
                    else "⭐ Premium" if not bool(c.is_high_risk) and fr["rental_rate"] >= 2.99
                    else "👍 Match")
            rc[5].markdown(note)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — GROWTH OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════
elif page == page_labels[2]:
    ph("💡 Growth Opportunities",
       "Actionable strategies to grow revenue, retain champions, and re-engage lost customers.")

    # ── OPP 1: BUNDLES ────────────────────────────────────────────────────
    sl("OPPORTUNITY 1 — BUNDLE PROMOTIONS")
    cc("Genre Co-Rental Heatmap",
       "How many customers rented both genres. Darker = stronger bundling candidate.")
    cust_genres = rentals.groupby(["customer_id", "category"]).size().reset_index(name="c")
    pivot = cust_genres.pivot_table(index="customer_id", columns="category",
                                    values="c", fill_value=0)
    pivot_bool = (pivot > 0).astype(int)
    gl = list(pivot_bool.columns)
    co_m = pivot_bool.T.dot(pivot_bool)
    np.fill_diagonal(co_m.values, 0)
    co_df = pd.DataFrame(co_m.values, index=gl, columns=gl)
    fig = px.imshow(co_df, color_continuous_scale=["#f0f9ff", C["blue"]],
                    labels=dict(color="Shared Customers"), aspect="auto", text_auto=True)
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter", size=10), margin=dict(l=8, r=8, t=10, b=8))
    st.plotly_chart(fig, use_container_width=True)

    pairs = sorted(
        [(gl[i], gl[j], int(co_df.iloc[i, j]))
         for i in range(len(gl)) for j in range(i+1, len(gl))],
        key=lambda x: -x[2],
    )[:5]
    sl("TOP 5 BUNDLE PAIRS")
    bc = st.columns(5)
    for col, (g1, g2, n) in zip(bc, pairs):
        pct = n / len(profiles) * 100
        col.markdown(
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:9px;'
            f'padding:11px 9px;text-align:center"><div style="font-size:1.1rem">📦</div>'
            f'<div style="font-weight:600;font-size:0.79rem;color:#1e3a8a">'
            f'{g1}<br>+<br>{g2}</div>'
            f'<div style="font-size:1.05rem;font-weight:800;color:{C["blue"]}">{n:,}</div>'
            f'<div style="font-size:0.7rem;color:#6b7280">customers ({pct:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )
    co(f"<b>Action:</b> Launch a 2+1 combo deal for "
       f"<b>{pairs[0][0]} + {pairs[0][1]}</b> "
       f"({pairs[0][2]:,} customers share both genres — {pairs[0][2]/len(profiles)*100:.0f}% of base).",
       "green")

    st.markdown("---")

    # ── OPP 2: RE-ENGAGEMENT — REPLACED CHARTS ────────────────────────────
    sl("OPPORTUNITY 2 — RE-ENGAGEMENT")

    # Chart A: Average days since last rental per segment (simple grouped bar — easy to read)
    re1, re2 = st.columns(2)
    with re1:
        cc("How Long Since Each Segment Last Rented?",
           "Average days since last rental per RFM segment. "
           "Higher = more lapsed. Target the highest-bar segments first.")
        seg_recency = (
            profiles.groupby("rfm_segment")["days_since_last_rental"]
            .mean().reset_index().sort_values("days_since_last_rental", ascending=False)
        )
        fig = px.bar(
            seg_recency,
            x="rfm_segment", y="days_since_last_rental",
            color="rfm_segment", color_discrete_map=SEG_COLORS,
            text=seg_recency["days_since_last_rental"].apply(lambda v: f"{v:.0f} days"),
            labels={"rfm_segment": "Segment",
                    "days_since_last_rental": "Avg Days Since Last Rental"},
            category_orders={"rfm_segment": ["Lost", "At Risk", "Loyal", "Champions"]},
        )
        fig.update_traces(textposition="outside")
        cf(fig, 280)
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        lost_days = float(seg_recency[seg_recency["rfm_segment"] == "Lost"]
                          ["days_since_last_rental"].values[0]) \
            if "Lost" in seg_recency["rfm_segment"].values else 0
        co(f"<b>Lost</b> customers haven't rented for an average of "
           f"<b>{lost_days:.0f} days</b>. Send a re-activation offer now before "
           "they become permanently inactive.", "red")

    with re2:
        cc("Revenue Potential by Segment",
           "Avg spending per customer in each segment — shows how much is at stake.")
        seg_spend = (
            profiles.groupby("rfm_segment")["total_spending"]
            .mean().reset_index().sort_values("total_spending", ascending=False)
        )
        # Add count annotation
        seg_cnt = profiles.groupby("rfm_segment").size().reset_index(name="count")
        seg_spend = seg_spend.merge(seg_cnt, on="rfm_segment")
        fig = px.bar(
            seg_spend,
            x="rfm_segment", y="total_spending",
            color="rfm_segment", color_discrete_map=SEG_COLORS,
            text=seg_spend["total_spending"].apply(lambda v: f"${v:.0f}"),
            labels={"rfm_segment": "Segment", "total_spending": "Avg Spending ($)"},
            category_orders={"rfm_segment": ["Champions", "Loyal", "At Risk", "Lost"]},
        )
        fig.update_traces(textposition="outside")
        cf(fig, 280)
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        champ_spend = float(seg_spend[seg_spend["rfm_segment"] == "Champions"]
                            ["total_spending"].values[0]) \
            if "Champions" in seg_spend["rfm_segment"].values else 0
        co(f"Champions spend <b>${champ_spend:.0f}</b> on average — "
           "the highest of all segments. Protecting this group is the highest-ROI action.",
           "green")

    sl("ACTION RECOMMENDATIONS")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        ac("🏆", "Reward Champions",
           f"{int((profiles['rfm_segment']=='Champions').sum())} customers. "
           "VIP early access, loyalty card, exclusive new-release previews. "
           "Goal: maintain engagement and prevent churn.")
    with ac2:
        ac("⚠️", "Re-engage At-Risk",
           f"{int((profiles['rfm_segment']=='At Risk').sum())} customers. "
           "'We miss you — 15% off your next rental.' "
           "Personalise with their favourite genres.")
    with ac3:
        ac("💤", "Win Back Lost",
           f"{int((profiles['rfm_segment']=='Lost').sum())} customers. "
           "Bold offer: one free rental. Focus on past high-spenders first.")

    st.markdown("---")

    # ── OPP 3: REGIONAL — MAP + BIGGER CHART + FIXED COUNTRY CHART ──────
    sl("OPPORTUNITY 3 — REGIONAL TARGETING")

    ctry_summary = get_country_summary(profiles)
    pivot_cg     = get_country_genre_pivot(rentals, profiles)

    # Customer map (scatter_geo)
    cc("Customer Distribution by Country",
       "Bubble size = number of customers. Color = average spending.")
    # get lat/lon centroids for scatter_geo — use country name
    fig_map = px.scatter_geo(
        ctry_summary,
        locations="country",
        locationmode="country names",
        size="customers",
        color="avg_spending",
        color_continuous_scale=["#dbeafe", C["indigo"]],
        hover_name="country",
        hover_data={
            "customers": True,
            "avg_spending": ":.2f",
            "avg_late": ":.1%",
        },
        size_max=40,
        projection="natural earth",
        labels={"customers": "Customers", "avg_spending": "Avg Spending ($)",
                "avg_late": "Avg Late Rate"},
    )
    fig_map.update_layout(
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        geo=dict(
            showframe=False, showcoastlines=True,
            coastlinecolor="#d1d5db",
            showland=True, landcolor="#f9fafb",
            showocean=True, oceancolor="#eff6ff",
            showcountries=True, countrycolor="#e5e7eb",
            bgcolor="white",
        ),
        coloraxis_colorbar=dict(title="Avg Spending ($)", tickprefix="$"),
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter", size=11),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # Country bar chart — BIGGER
    cc("Customers & Avg Spending by Country",
       "Bar = number of customers. Line = average spending per customer.")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ctry_summary["country"], y=ctry_summary["customers"],
        name="Customers", marker_color=C["blue"],
        text=ctry_summary["customers"], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=ctry_summary["country"], y=ctry_summary["avg_spending"],
        name="Avg Spending ($)", mode="lines+markers",
        line=dict(color=C["amber"], width=2.5),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig.update_layout(
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=11),
        margin=dict(l=8, r=8, t=10, b=80),
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(tickangle=-40, showgrid=True, gridcolor="#f3f4f6", title=""),
        yaxis=dict(title="Number of Customers", showgrid=True, gridcolor="#f3f4f6"),
        yaxis2=dict(title="Avg Spending ($)", overlaying="y", side="right",
                    showgrid=False),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top 3 per country genre — REPLACED heatmap with readable grouped bar
    st.markdown("---")
    cc("Top Genre per Country",
       "The single most-rented genre in each country — easy to act on for regional campaigns.")
    top_genre_per_country = (
        pivot_cg.idxmax(axis=1).reset_index()
        .rename(columns={0: "top_genre"})
    )
    top_genre_per_country["share"] = [
        float(pivot_cg.loc[row["country"], row["top_genre"]])
        for _, row in top_genre_per_country.iterrows()
    ]
    top_genre_per_country = top_genre_per_country.merge(
        ctry_summary[["country", "customers"]], on="country"
    ).sort_values("customers", ascending=False)

    fig = px.bar(
        top_genre_per_country,
        x="country", y="share",
        color="top_genre",
        text=top_genre_per_country.apply(
            lambda r: f"{r['top_genre']}\n{r['share']:.0f}%", axis=1),
        labels={"country": "Country", "share": "Genre Share (%)", "top_genre": "Top Genre"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(textposition="outside")
    cf(fig, 340)
    fig.update_layout(xaxis=dict(tickangle=-40), yaxis_title="Share of Rentals (%)")
    st.plotly_chart(fig, use_container_width=True)

    co("<b>Action:</b> Stock more of each country's dominant genre and feature it "
       "in region-specific email campaigns for higher conversion.", "green")
    co("💡 <b>Next →</b> Go to <b>Predictions & Risk</b> to forecast individual "
       "customer behaviour.", "blue")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIONS & RISK
# ══════════════════════════════════════════════════════════════════════════
elif page == page_labels[3]:
    ph("🔮 Predictions & Risk",
       "Forecast customer behaviour with ML models. "
       "Choose a prediction type below.")

    pred_type = st.selectbox(
        "What do you want to predict?",
        [
            "🔴 Late Return Risk — Will this customer return DVDs late?",
            "🎭 Next Rental Category — What genre will a specific customer rent next?",
            "📉 Churn Risk — Which customers are about to stop renting?",
        ],
        help="Select a prediction type, then complete the form.",
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════
    # A — LATE RETURN RISK
    # ════════════════════════════════════════════════════
    if "Late Return Risk" in pred_type:
        high_r   = int(profiles["is_high_risk"].sum())
        low_r    = len(profiles) - high_r
        late_pct = float(rentals["is_late_return"].fillna(False).sum() /
                         rentals["return_date"].notna().sum() * 100)

        co(f"<b>{high_r:,}</b> customers ({high_r/len(profiles)*100:.1f}%) are currently "
           f"high-risk. Overall late return rate: <b>{late_pct:.1f}%</b>.", "amber")

        sl("LATE RETURN CONTEXT")
        ov1, ov2, ov3 = st.columns(3)
        with ov1:
            cc("Risk Split")
            fig = go.Figure(go.Pie(
                labels=["High Risk", "Low Risk"], values=[high_r, low_r],
                hole=0.6, marker_colors=[C["red"], C["green"]], textinfo="label+percent"))
            fig.update_layout(height=220, showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                              font=dict(family="Inter", size=12),
                              annotations=[dict(text=f"<b>{len(profiles):,}</b>",
                                                x=0.5, y=0.5, font_size=14, showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
        with ov2:
            cat_df = get_category_stats()
            cat_df["late_rate"] = cat_df["late_count"] / cat_df["rentals"] * 100
            cc("Late Rate by Genre")
            fig = px.bar(cat_df.sort_values("late_rate", ascending=True),
                         x="late_rate", y="category", orientation="h",
                         color="late_rate", color_continuous_scale=["#fef9c3", C["red"]],
                         labels={"late_rate": "Late Rate (%)", "category": ""},
                         text=cat_df.sort_values("late_rate")["late_rate"]
                                    .apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            cf(fig, 220)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with ov3:
            cc("High-Risk % by Segment")
            rs = profiles.groupby("rfm_segment")["is_high_risk"].mean().reset_index()
            rs["pct"] = rs["is_high_risk"] * 100
            fig = px.bar(rs.sort_values("pct"),
                         x="pct", y="rfm_segment", orientation="h",
                         color="rfm_segment", color_discrete_map=SEG_COLORS,
                         labels={"pct": "% High Risk", "rfm_segment": ""},
                         text=rs.sort_values("pct")["pct"].apply(lambda v: f"{v:.0f}%"))
            fig.update_traces(textposition="outside")
            cf(fig, 220)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        sl("TOP 10 HIGHEST-RISK CUSTOMERS")
        hr_df = (
            profiles[profiles["is_high_risk"]]
            [["customer_id", "full_name", "city", "country", "total_rentals",
              "late_return_count", "late_return_rate", "avg_days_overdue",
              "total_spending", "rfm_segment", "risk_score"]]
            .sort_values("risk_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        hr_df["late_return_rate"] = (hr_df["late_return_rate"] * 100).round(1)
        hr_df["avg_days_overdue"] = hr_df["avg_days_overdue"].round(1)
        hr_df["risk_score"]       = hr_df["risk_score"].round(3)
        st.dataframe(hr_df.rename(columns={
            "customer_id": "ID", "full_name": "Name", "city": "City",
            "country": "Country", "total_rentals": "Rentals",
            "late_return_count": "Late Returns", "late_return_rate": "Late Rate (%)",
            "avg_days_overdue": "Avg Overdue (d)", "total_spending": "Spent ($)",
            "rfm_segment": "Segment", "risk_score": "Risk Score"}),
            use_container_width=True, height=380)

        st.markdown("---")
        sl("PREDICT RISK FOR A NEW CUSTOMER")
        co("Enter data for a new customer not yet in the system. "
           "Fields marked ★ are <b>calculated automatically</b> from your inputs.", "blue")

        sel_model = st.selectbox("Prediction model", list(FEATURE_SETS.keys()),
                                 help="Random Forest/Gradient Boosting = more accurate. "
                                      "Logistic Regression = easiest to explain.")
        minfo = FEATURE_SETS[sel_model]
        st.markdown(
            f"<div style='border:1px solid {minfo['color']}44;border-radius:7px;"
            f"padding:8px 12px;background:{minfo['color']}0a;font-size:0.81rem;"
            f"margin-bottom:8px'><b style='color:{minfo['color']}'>{sel_model}:</b> "
            f"{minfo['description']}</div>",
            unsafe_allow_html=True,
        )

# Search customer
        ns1, ns2 = st.columns([3, 1])
        with ns1:
            rq = st.text_input("Search customer name", placeholder="e.g. Mary Smith", key="rq")
        with ns2:
            rid = st.text_input("Customer ID", placeholder="ID", key="rid")

        rflt = profiles.copy()
        if rq.strip():
            rflt = rflt[rflt["full_name"].str.contains(rq.strip(), case=False, na=False)]
        if rid.strip().isdigit():
            rflt = rflt[rflt["customer_id"] == int(rid.strip())]
        rflt = rflt.head(10)

        if not rflt.empty:
            sel_rid = st.selectbox(
                "Select customer",
                rflt["customer_id"].tolist(),
                format_func=lambda i: f"#{i} — {rflt[rflt.customer_id==i]['full_name'].values[0]}",
                key="rsel",
            )

            cr = profiles[profiles["customer_id"] == sel_rid].iloc[0]

            # Hitung genre_diversity otomatis
            cust_genres = rentals[rentals["customer_id"] == sel_rid]["category"].nunique()
            total_r = int(cr.total_rentals)
            genre_div = round(cust_genres / total_r, 2) if total_r > 0 else 0

            # Tampilkan data auto-filled
            st.markdown("---")
            sl("CUSTOMER DATA — AUTO FILLED FROM DATABASE")
            col1, col2 = st.columns(2)
            with col1:
                ro("📦 Total Rentals", int(cr.total_rentals))
                ro("🔴 Late Return Count", int(cr.late_return_count))
                ro("💵 Total Spending ($)", f"${float(cr.total_spending):.2f}")
                ro("🕐 Days Since Last Rental", int(cr.days_since_last_rental))
            with col2:
                ro("📅 Avg Days Between Rentals", f"{float(cr.avg_rental_gap_days):.1f} days")
                ro("⏳ Avg Days Overdue", f"{float(cr.avg_days_overdue):.2f} days")
                ro("★ Late Return Rate (auto)", f"{float(cr.late_return_rate)*100:.1f}%")
                ro("★ Genre Diversity (auto)", genre_div)
                ro("★ Avg Spend per Rental (auto)", f"${float(cr.avg_spend_per_rental):.2f}")

            submitted = st.button("⚠️ Assess Risk", type="primary", use_container_width=True)

            if submitted:
                result = predict_manual(sel_model, trained_models, {
                    "total_rentals":          int(cr.total_rentals),
                    "avg_days_overdue":       float(cr.avg_days_overdue),
                    "late_return_count":      int(cr.late_return_count),
                    "late_return_rate":       float(cr.late_return_rate),
                    "total_spending":         float(cr.total_spending),
                    "avg_rental_gap_days":    float(cr.avg_rental_gap_days),
                    "days_since_last_rental": int(cr.days_since_last_rental),
                    "genre_diversity":        genre_div,
                })
                prob_pct = result["probability"] * 100
                color    = C["red"] if result["predicted_risk"] else C["green"]
                sl("PREDICTION RESULT")
                res1, res2 = st.columns([1, 2])
                with res1:
                    label = "High Risk" if result["predicted_risk"] else "Low Risk"
                    icon  = "⚠️" if result["predicted_risk"] else "✅"
                    st.markdown(
                        f'<div style="border:2px solid {color};border-radius:12px;'
                        f'padding:20px 14px;text-align:center;background:{color}0d">'
                        f'<div style="font-size:2rem">{icon}</div>'
                        f'<div style="font-weight:700;font-size:1.1rem;color:{color}">{label}</div>'
                        f'<div style="font-size:2.5rem;font-weight:800;color:{color};line-height:1">'
                        f'{prob_pct:.1f}%</div>'
                        f'<div style="font-size:0.74rem;color:#6b7280">probability</div>'
                        f'<div style="background:#e5e7eb;border-radius:99px;height:5px;margin-top:10px">'
                        f'<div style="background:{color};width:{prob_pct:.0f}%;height:5px;'
                        f'border-radius:99px"></div></div></div>',
                        unsafe_allow_html=True,
                    )
                with res2:
                    cc("What's driving this score?",
                       "Each bar = contribution of that factor to the prediction.")
                    contribs = result["contributions"]
                    cdf = pd.DataFrame({
                        "Factor": list(contribs.keys()),
                        "Contribution (%)": list(contribs.values()),
                    }).sort_values("Contribution (%)", ascending=True)
                    fig = px.bar(cdf, x="Contribution (%)", y="Factor", orientation="h",
                                 color="Contribution (%)",
                                 color_continuous_scale=["#dbeafe", C["indigo"]],
                                 text=cdf["Contribution (%)"].apply(lambda v: f"{v:.1f}%"))
                    fig.update_traces(textposition="outside")
                    cf(fig, 220)
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Stakeholder insights
                st.markdown("---")
                sl("RECOMMENDED ACTIONS & BUSINESS INSIGHTS")
                top_f = max(contribs, key=contribs.get)

                if result["predicted_risk"]:
                    i1, i2, i3 = st.columns(3)
                    with i1:
                        insight_card("imm", "🚨 Immediate Action",
                            "Send Return Reminder",
                            f"Top driver: <b>{top_f}</b> ({contribs[top_f]:.1f}%). "
                            "Enable automated SMS/email 1 day before due date for this customer. "
                            "Cost: near-zero. Expected outcome: reduce overdue days immediately.")
                    with i2:
                        insight_card("short", "📅 Short-Term (1–4 weeks)",
                            "Offer On-Time Incentive",
                            "Reward the next 3 on-time returns with 10% off the following rental. "
                            "Behavioural nudge — positive reinforcement is more effective than penalties.")
                    with i3:
                        insight_card("long", "📈 Long-Term (1–3 months)",
                            "Limit Concurrent Rentals",
                            "Cap at 2 active rentals until behaviour improves. "
                            "Review status monthly. If late rate drops below 20%, restore full access.")
                else:
                    i1, i2, i3 = st.columns(3)
                    with i1:
                        insight_card("long", "✅ Immediate Opportunity",
                            "Offer Premium or Extended Plan",
                            "This customer is reliable. Offer longer rental periods "
                            "(7+ days) and premium-rate titles. Low risk = high trust = upsell opportunity.")
                    with i2:
                        insight_card("long", "📅 Short-Term",
                            "Enrol in Loyalty Programme",
                            "Reliable customers enrolled in loyalty schemes have "
                            "significantly higher lifetime value. Target for VIP membership invitation.")
                    with i3:
                        insight_card("long", "📈 Long-Term",
                            "Use as Brand Advocate",
                            "Low-risk, high-frequency customers are ideal referral sources. "
                            "A 'refer a friend' programme with a shared discount can expand your base.")

    # ════════════════════════════════════════════════════
    # B — NEXT RENTAL CATEGORY (PER CUSTOMER)
    # ════════════════════════════════════════════════════
    elif "Next Rental Category" in pred_type:
        sl("NEXT RENTAL CATEGORY — CUSTOMER-SPECIFIC")
        co("Select a specific customer. The prediction is based on <b>their personal "
           "rental history</b> — what they rented last, and what similar customers "
           "rented next.", "blue")

        # Customer search
        ns1, ns2 = st.columns([3, 1])
        with ns1:
            nq = st.text_input("Search customer name", placeholder="e.g. Mary Smith",
                               label_visibility="collapsed", key="nq")
        with ns2:
            nid = st.text_input("Customer ID", placeholder="ID",
                                label_visibility="collapsed", key="nid")

        nflt = profiles.copy()
        if nq.strip():
            nflt = nflt[nflt["full_name"].str.contains(nq.strip(), case=False, na=False)]
        if nid.strip().isdigit():
            nflt = nflt[nflt["customer_id"] == int(nid.strip())]
        nflt = nflt.head(10)

        if nflt.empty:
            st.info("Type a name or ID to find a customer.")
            st.stop()

        sel_cid = st.selectbox(
            "Select customer",
            nflt["customer_id"].tolist(),
            format_func=lambda i: f"#{i} — {nflt[nflt.customer_id==i]['full_name'].values[0]}",
            key="nsel",
        )
        c_row   = profiles[profiles["customer_id"] == sel_cid].iloc[0]
        c_rent  = get_customer_rentals(sel_cid, rentals)
        genres  = get_genre_preferences(sel_cid, rentals)

        st.markdown(
            pill(f"ID #{sel_cid}") + pill(f"{c_row.city}, {c_row.country}") +
            sb(c_row.rfm_segment) + " " + rb(bool(c_row.is_high_risk)),
            unsafe_allow_html=True,
        )
        st.markdown("")

        if c_rent.empty:
            st.warning("No rental history found for this customer.")
            st.stop()

        # Their most recent rental
        last_rental = c_rent.sort_values("rental_date", ascending=False).iloc[0]
        last_cat    = last_rental["category"]
        last_film   = last_rental["title"]
        last_date   = last_rental["rental_date"].strftime("%d %b %Y")

        co(f"Most recent rental: <b>{last_film}</b> ({last_cat}) on <b>{last_date}</b>. "
           f"Prediction is based on this genre as the starting point.", "blue")

        # Genre transition probabilities from their last category
        trans_for_cat = trans_mx[trans_mx["category"] == last_cat].copy()
        trans_for_cat = trans_for_cat.sort_values("prob", ascending=False).head(8)

        # Boost probability for genres the customer already likes
        genre_rank = {r["category"]: (len(genres) - i) for i, r in genres.iterrows()}
        max_rank   = max(genre_rank.values()) if genre_rank else 1
        trans_for_cat["personal_boost"] = (
            trans_for_cat["next_category"].map(genre_rank).fillna(0) / max_rank * 0.15
        )
        trans_for_cat["adjusted_prob"] = (
            (trans_for_cat["prob"] + trans_for_cat["personal_boost"])
        )
        trans_for_cat["adjusted_prob"] = (
            trans_for_cat["adjusted_prob"] / trans_for_cat["adjusted_prob"].sum()
        )
        trans_for_cat = trans_for_cat.sort_values("adjusted_prob", ascending=False)

        sl("PREDICTED NEXT GENRES")
        for _, row in trans_for_cat.head(6).iterrows():
            p    = float(row["adjusted_prob"]) * 100
            base = float(row["prob"]) * 100
            boost= p - base
            boost_label = (f" <span style='font-size:0.7rem;color:{C['green']}'>"
                           f"+{boost:.1f}% personal boost</span>"
                           if boost > 0.5 else "")
            st.markdown(
                f'<div class="pc">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div><b style="font-size:0.93rem">🎭 {row["next_category"]}</b>'
                f'{boost_label}</div>'
                f'<div style="font-size:1.1rem;font-weight:800;color:{C["blue"]}">'
                f'{p:.1f}%</div></div>'
                f'<div style="background:#e5e7eb;border-radius:99px;height:6px;margin:6px 0">'
                f'<div style="background:{C["blue"]};width:{min(int(p*4),100)}%;'
                f'height:6px;border-radius:99px"></div></div>'
                f'<div style="font-size:0.73rem;color:#9ca3af">'
                f'Based on {int(row["count"]):,} sequences · '
                f'Base probability: {base:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        top_next = trans_for_cat.iloc[0]["next_category"]
        top_prob = float(trans_for_cat.iloc[0]["adjusted_prob"]) * 100

        # Film recs from the predicted category, filtered to not-yet-rented
        rented_ids = set(c_rent["film_id"].tolist())
        recs_next  = films[
            (films["category"] == top_next) &
            (~films["film_id"].isin(rented_ids))
        ].sort_values("rental_rate", ascending=False).head(6)

        st.markdown("---")
        sl(f"FILM RECOMMENDATIONS IN '{top_next.upper()}'")
        cc(f"Top Unrented Films in {top_next}",
           f"Recommended based on {top_prob:.0f}% predicted interest in {top_next}.")

        rh = st.columns([2.5, 0.7, 1, 1])
        for h, lbl in zip(rh, ["Film Title", "Rating", "Rental Period", "Price"]):
            h.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)
        st.divider()
        for _, fr in recs_next.iterrows():
            rc = st.columns([2.5, 0.7, 1, 1])
            rc[0].markdown(f"**{fr['title']}**")
            rc[1].markdown(fr["rating"])
            rc[2].markdown(f"{int(fr['rental_duration'])} days")
            rc[3].markdown(f"${fr['rental_rate']:.2f}")

        # Stakeholder insights
        st.markdown("---")
        sl("BUSINESS INSIGHTS FROM THIS PREDICTION")
        bi1, bi2, bi3 = st.columns(3)
        with bi1:
            insight_card("imm", "🚨 Immediate",
                "Personalised Recommendation Email",
                f"Send {c_row.full_name} an email today featuring new "
                f"<b>{top_next}</b> titles. Personalised outreach has 3–6× "
                "higher click-through than generic campaigns.")
        with bi2:
            insight_card("short", "📅 Short-Term",
                "In-Store / App Cross-Sell",
                f"When {c_row.full_name} returns a <b>{last_cat}</b> film, "
                f"automatically surface <b>{top_next}</b> suggestions at checkout. "
                "Contextual prompts increase basket size by 15–25% on average.")
        with bi3:
            insight_card("long", "📈 Long-Term",
                "Genre Affinity Programme",
                f"Customers who frequently transition {last_cat} → {top_next} "
                "form a natural affinity cluster. Build a segment-level campaign "
                "targeting all such customers for sustained revenue growth.")

    # ════════════════════════════════════════════════════
    # C — CHURN RISK  (replaces Spending Tier)
    # ════════════════════════════════════════════════════
    elif "Churn Risk" in pred_type:
        sl("CHURN RISK ANALYSIS")
        co("Churn risk measures how likely a customer is to <b>stop renting entirely</b>. "
           "It is calculated from recency (days since last rental), frequency decline, "
           "and segment status — not just late returns.", "amber")

        # Churn score: simple composite
        max_gap = float(profiles["days_since_last_rental"].max())
        profiles_c = profiles.copy()
        profiles_c["churn_score"] = (
            (profiles_c["days_since_last_rental"] / max_gap * 0.6) +
            (1 - profiles_c["total_rentals"] /
             profiles_c["total_rentals"].max() * 0.25) +
            (profiles_c["rfm_segment"].map(
                {"Lost": 1.0, "At Risk": 0.6, "Loyal": 0.2, "Champions": 0.0}
            ) * 0.15)
        ).clip(0, 1).round(3)

        profiles_c["churn_risk_label"] = pd.cut(
            profiles_c["churn_score"],
            bins=[0, 0.33, 0.66, 1.01],
            labels=["Low Churn Risk", "Medium Churn Risk", "High Churn Risk"],
        )

        cr_counts = profiles_c["churn_risk_label"].value_counts()
        cr_rev    = profiles_c.groupby("churn_risk_label")["total_spending"].sum()

        cr1, cr2, cr3 = st.columns(3)
        for col, (lbl, color) in zip(
            [cr1, cr2, cr3],
            [("High Churn Risk", C["red"]),
             ("Medium Churn Risk", C["amber"]),
             ("Low Churn Risk", C["green"])],
        ):
            cnt = int(cr_counts.get(lbl, 0))
            rev = float(cr_rev.get(lbl, 0))
            col.metric(lbl, f"{cnt:,}", f"${rev:,.0f} revenue at stake",
                       delta_color="off")

        st.markdown("---")
        sl("CHURN SCORE DISTRIBUTION")
        ch1, ch2 = st.columns(2)
        with ch1:
            cc("Churn Risk by Segment",
               "What % of each RFM segment is high-churn risk?")
            seg_churn = (
                profiles_c.groupby("rfm_segment")["churn_score"]
                .mean().reset_index()
                .sort_values("churn_score", ascending=False)
            )
            fig = px.bar(
                seg_churn, x="rfm_segment", y="churn_score",
                color="rfm_segment", color_discrete_map=SEG_COLORS,
                text=seg_churn["churn_score"].apply(lambda v: f"{v:.2f}"),
                labels={"rfm_segment": "Segment", "churn_score": "Avg Churn Score (0–1)"},
            )
            fig.update_traces(textposition="outside")
            cf(fig, 280)
            fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            cc("Revenue at Risk by Country",
               "Total spending of high-churn customers per country.")
            hi_churn = profiles_c[profiles_c["churn_risk_label"] == "High Churn Risk"]
            rev_at_risk = (
                hi_churn.groupby("country")["total_spending"]
                .sum().reset_index()
                .sort_values("total_spending", ascending=True)
                .tail(15)
            )
            fig = px.bar(
                rev_at_risk, x="total_spending", y="country", orientation="h",
                color="total_spending",
                color_continuous_scale=["#fef9c3", C["red"]],
                labels={"total_spending": "Revenue at Risk ($)", "country": ""},
                text=rev_at_risk["total_spending"].apply(lambda v: f"${v:,.0f}"),
            )
            fig.update_traces(textposition="outside")
            cf(fig, 280)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        sl("TOP 10 CUSTOMERS MOST LIKELY TO CHURN")
        top_churn = (
            profiles_c[["customer_id", "full_name", "city", "country",
                         "rfm_segment", "days_since_last_rental",
                         "total_rentals", "total_spending",
                         "churn_score", "churn_risk_label"]]
            .sort_values("churn_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_churn["churn_score"] = top_churn["churn_score"].round(3)
        st.dataframe(top_churn.rename(columns={
            "customer_id": "ID", "full_name": "Name", "city": "City",
            "country": "Country", "rfm_segment": "Segment",
            "days_since_last_rental": "Days Since Last Rental",
            "total_rentals": "Total Rentals", "total_spending": "Total Spent ($)",
            "churn_score": "Churn Score", "churn_risk_label": "Risk Level"}),
            use_container_width=True, height=380)

        total_at_risk_rev = float(
            profiles_c[profiles_c["churn_risk_label"] == "High Churn Risk"]["total_spending"].sum()
        )
        st.markdown("---")
        sl("BUSINESS INSIGHTS FROM CHURN ANALYSIS")
        bi1, bi2, bi3 = st.columns(3)
        with bi1:
            insight_card("imm", "🚨 Immediate Action",
                "Contact Top 10 Churn Risks Today",
                f"The top 10 customers represent significant revenue. "
                f"A personal outreach (phone or email) with a special offer "
                "has the highest chance of recovery for high-value churners.")
        with bi2:
            insight_card("short", "📅 Short-Term (2–4 weeks)",
                "Launch Re-activation Campaign",
                f"<b>${total_at_risk_rev:,.0f}</b> in revenue is at risk from "
                "high-churn customers. A targeted email campaign with a 'come back' "
                "offer can recover 15–30% of lapsed revenue.")
        with bi3:
            insight_card("long", "📈 Long-Term (1–3 months)",
                "Build Churn Early-Warning System",
                "Automate monthly churn scoring. Flag any customer whose recency "
                "doubles (e.g. from 14 to 28 days gap) for proactive outreach. "
                "Prevention costs 5× less than re-acquisition.")

    # Model reference
    st.markdown("---")
    with st.expander("📊 Model performance details"):
        rows = []
        for mname, mdata in trained_models.items():
            m = mdata["metrics"]
            rows.append({"Model": mname, "Features": len(mdata["feature_cols"]),
                         "Accuracy": f"{m['accuracy']:.1%}", "AUC": f"{m['roc_auc']:.3f}",
                         "F1": f"{m['f1']:.3f}",
                         "CV AUC": f"{m['cv_auc_mean']:.3f} ± {m['cv_acc_std']:.3f}"})
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        best = max(trained_models.items(), key=lambda x: x[1]["metrics"]["cv_auc_mean"])
        st.markdown(f"**Best model:** {best[0]} (CV AUC = {best[1]['metrics']['cv_auc_mean']:.3f})")