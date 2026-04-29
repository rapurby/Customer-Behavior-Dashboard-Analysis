"""
db.py — PostgreSQL connection untuk dvdrental database.
Edit bagian DB_CONFIG sesuai settingan PostgreSQL kamu.
"""

import psycopg2
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════
# KONFIGURASI — edit sesuai setup PostgreSQL kamu
# ══════════════════════════════════════════════════════════
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "dvdrental",
    "user":     "postgres",
    "password": os.getenv("DB_PASSWORD"),  
}


def get_connection():
    """Buat koneksi ke PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


def query(sql: str, params=None) -> pd.DataFrame:
    """Jalankan SELECT query dan return sebagai DataFrame."""
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=params)


def test_connection() -> bool:
    """Cek apakah koneksi berhasil."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception as e:
        return False, str(e)