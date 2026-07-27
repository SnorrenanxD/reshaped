# app.py
# Streamlit chat UI for the SMS assistant.
import json
import streamlit as st
from src.router import handle_query, flatten_answer
from src.llm import GEMINI_AVAILABLE, GEMINI_MODEL, OLLAMA_MODEL
from src.format import format_section_text

st.set_page_config(page_title="SMS Assistant", page_icon="⚓")

USER_AVATAR = "🧑‍✈️"
ASSISTANT_AVATAR = "⚓"
PHASE_COLOR = {"IMMEDIATE ACTIONS": "red", "CONDITIONAL": "orange", "REPORTING": "blue"}


@st.cache_resource
def get_chunks():
    """Ingested manual sections, loaded once per server process."""
    with open("data/processed/sections.json") as f:
        return json.load(f)


chunks = get_chunks()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("active_chunk", None)  # section a follow-up question continues from
st.session_state.setdefault("sections_consulted", set())
st.session_state.setdefault("last_model", None)


def badge_for_responsibility(text: str) -> str:
    """Master gets a red badge (accountable party); everyone else gets gray."""
    return "red" if "master" in text.lower() else "gray"


def render_sections_picker(sections: list[dict], key: str):
    """Pills to pick a cited section, expanding its unaltered manual text below."""
    if not sections:
        return
    if len(sections) == 1:
        st.caption("Grounded in — open the section below to read the unaltered manual text")
    else:
        st.caption(f"Grounded in {len(sections)} sections — open one to read the unaltered manual text")

    labels = [f"§{s['id']} {s['title']}" for s in sections]
    picked = st.pills("Sections", labels, key=key, label_visibility="collapsed")
    if picked:
        section = sections[labels.index(picked)]
        with st.expander(f"Full section text — §{section['id']} {section['title']}", expanded=True):
            st.caption(
                "Unaltered text as ingested from `Case Manual marineops_sms.pdf` · "
                "nothing below is model-generated"
            )
            st.markdown(format_section_text(section["text"]))


def render_answer(answer: dict, key: str):
    """Renders one assistant turn — plain text, or a full checkable workflow."""
    chunk = answer.get("chunk")

    if answer["type"] != "workflow":
        st.markdown(answer["text"])
    else:
        if answer["title"]:
            st.subheader(answer["title"])
        if answer["subtitle"]:
            st.caption(answer["subtitle"])

        if chunk and answer["confidence"] and answer["confidence"] != "high":
            st.warning(
                f"§{chunk['id']} was matched with **{answer['confidence']}** confidence — "
                "read it yourself before relying on the steps below. Whether it applies here "
                "is a judgment call, not this assistant's.",
                icon=":material/warning:",
            )

        if answer["callout"]:
            st.info(answer["callout"], icon=":material/info:")

        immediate_total = immediate_done = 0
        for p_idx, phase in enumerate(answer["phases"]):
            label = phase.get("label") or ""
            st.markdown(f":{PHASE_COLOR.get(label, 'gray')}-badge[{label}]")
            if phase.get("note"):
                st.caption(phase["note"])

            for s_idx, step in enumerate(phase.get("steps", [])):
                label_md = step.get("text", "")
                if step.get("responsibility"):
                    label_md += f"  :{badge_for_responsibility(step['responsibility'])}-badge[{step['responsibility']}]"
                if step.get("section_id"):
                    label_md += f"  :blue-badge[§{step['section_id']}]"
                checked = st.checkbox(label_md, key=f"{key}_p{p_idx}_s{s_idx}")
                if label == "IMMEDIATE ACTIONS":
                    immediate_total += 1
                    immediate_done += checked

        if immediate_total:
            st.progress(
                immediate_done / immediate_total,
                text=f"Immediate actions completed: {immediate_done} of {immediate_total}",
            )

        sections = ([chunk] if chunk else []) + answer.get("secondary", [])
        if sections:
            st.divider()
            render_sections_picker(sections, key=f"{key}_pills")

    model = answer.get("model")
    if model:
        color = "green" if model == GEMINI_MODEL else "gray"
        footer = [f":{color}-badge[● {model}]"]
        details = []
        if chunk and answer.get("confidence"):
            details.append(f"§{chunk['id']} matched with {answer['confidence']} confidence")
        n_sections = (1 if chunk else 0) + len(answer.get("secondary", []))
        if n_sections:
            details.append(f"{n_sections} section{'s' if n_sections != 1 else ''} retrieved")
        elif not chunk:
            details.append("no section retrieved")
        if answer.get("verified"):
            details.append("verified by re-read")
        if answer.get("elapsed") is not None:
            details.append(f"{answer['elapsed']:.1f} s")
        if details:
            footer.append(f":small[{' · '.join(details)}]")
        st.markdown("  ".join(footer))


def render_sidebar():
    """Model status, session history, and sections consulted so far."""
    with st.sidebar:
        st.title("SMS Assistant")
        st.caption(
            "Grounded in `marineops_sms.pdf` — VIMS Marine Operations Safety Management Manual"
        )
        st.divider()

        st.markdown("**Model**")
        last_model = st.session_state.last_model
        if last_model:
            color = "green" if last_model == GEMINI_MODEL else "gray"
            st.markdown(f":{color}-badge[● {last_model}]")
            if last_model != GEMINI_MODEL:
                cause = "unreachable" if GEMINI_AVAILABLE else "not configured"
                st.caption(f"`{GEMINI_MODEL}` {cause} — running on the local fallback.")
        elif GEMINI_AVAILABLE:
            st.caption(f"`{GEMINI_MODEL}` (API) · fallback `{OLLAMA_MODEL}` (local)")
        else:
            st.caption(f"No `GEMINI_API_KEY` — running on `{OLLAMA_MODEL}` (local).")

        user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if user_turns:
            st.divider()
            st.markdown("**This session**")
            for i, query in enumerate(user_turns):
                if i == len(user_turns) - 1:
                    with st.container(border=True):
                        st.markdown(query)
                else:
                    st.caption(query)

            if st.session_state.sections_consulted:
                cited = ", ".join(f"§{s}" for s in sorted(st.session_state.sections_consulted))
                st.caption(f"Sections consulted: {cited}")


# Replay the conversation so far (checkbox state persists via each message's key).
for i, message in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and message.get("answer"):
            render_answer(message["answer"], key=f"msg{i}")
        else:
            st.markdown(message["content"])

# Handle a new turn.
if prompt := st.chat_input("Describe the situation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Checking the manual..."):
            answer, used_chunk = handle_query(
                st.session_state.messages, chunks, st.session_state.active_chunk
            )
        render_answer(answer, key=f"msg{len(st.session_state.messages)}")

    # Track session-level state for the sidebar.
    st.session_state.active_chunk = used_chunk
    st.session_state.last_model = answer.get("model")
    for section in ([used_chunk] if used_chunk else []) + answer.get("secondary", []):
        st.session_state.sections_consulted.add(section["id"])
    # Store the flattened text (for LLM context on the next turn) alongside the
    # structured answer (for re-rendering the rich UI from history).
    st.session_state.messages.append(
        {"role": "assistant", "content": flatten_answer(answer), "answer": answer}
    )

render_sidebar()
