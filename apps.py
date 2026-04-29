import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import datetime

st.title("Streamlit Session 1")

engine = create_engine(
    
    "postgresql+psycopg2://postgres:day0805@localhost:5432/dvdrental"
)

query = "SELECT * FROM kpi_customer_monthly LIMIT 20"
df = pd.read_sql(text(query), engine)

st.dataframe(df)

import datetime

st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.text("Plain text")
st.write("Flexible display (text, dataframe, etc)")
st.markdown("**Bold text** or markdown")

st.dataframe(df)
st.table(df)
st.json(df.to_dict())

st.metric("Total Sales", 10000)
st.metric("Growth", "15%", "+2%")

name = st.text_input("Enter your name")
age = st.number_input("Enter age", min_value=0)

option = st.selectbox("Choose one", ["A", "B", "C"])
options = st.multiselect("Choose multiple", ["A", "B", "C"])
value = st.slider("Select value", 0, 100)

if name:
    st.write(f"Hello {name}")

st.button("this is button")

if st.button('click me'):
    st.write('Hello World!')

if st.button('Current Time'):
    st.write(datetime.datetime.now())

st.line_chart(df.select_dtypes(include='number'))
st.bar_chart(df.select_dtypes(include='number'))
st.area_chart(df.select_dtypes(include='number'))
st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("Sales Dashboard")
st.write("Monthly business performance overview")

with st.sidebar:
    st.header("Filters")
    month = st.selectbox("Month", ["Jan", "Feb", "Mar"])
    store = st.selectbox("Store", ["All", "Store 1", "Store 2"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Revenue", "120,000")
with col2:
    st.metric("Orders", "2,450")
with col3:
    st.metric("Customers", "980")
with col4:
    st.metric("Avg Order", "49")

left, right = st.columns([2, 1])

with left:
    st.subheader("Revenue Trend")
    st.write("Main chart here")

with right:
    st.subheader("Top Categories")
    st.write("Bar chart here")

tab1, tab2 = st.tabs(["Detail Table", "Insights"])

with tab1:
    st.write("Detailed data table here")

with tab2:
    st.write("Business insight or chatbot explanation here")