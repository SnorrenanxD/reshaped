# app.py
import json
import streamlit as st
from src.router import handle_query

st.title("SMS Assistant")


@st.cache_resource
def get_chunks():
    with open("data/processed/sections.json") as f:
        return json.load(f)


chunks = get_chunks()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_chunk" not in st.session_state:
    st.session_state.active_chunk = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe the situation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream, used_chunk = handle_query(
            st.session_state.messages, chunks, st.session_state.active_chunk
        )
        response_text = st.write_stream(response_stream)

    st.session_state.active_chunk = used_chunk
    st.session_state.messages.append({"role": "assistant", "content": response_text})