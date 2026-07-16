# app.py
# Streamlit app that provides a chat interface for the SMS Assistant.

import streamlit as st
from src.search import build_retriever
from src.llm import warm_up
from src.router import handle_query

st.title("SMS Assistant")


@st.cache_resource
def get_retriever():
    return build_retriever("data/processed/sections.json")


@st.cache_resource
def get_warmed_up_llm():
    warm_up()
    return True


retriever = get_retriever()
get_warmed_up_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe the situation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(handle_query(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response})