import streamlit as st

openai_key = st.secrets["OPENAI_API_KEY"]
alpha_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

st.write("Keys loaded successfully!")
st.write("API key loaded successfully!")
