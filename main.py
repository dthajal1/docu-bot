from typing import Set

from backend.core import run_llm
import streamlit as st
# streamlit_chat is a custom component that quickly adds a chat UI to your Streamlit app.
from streamlit_chat import message

st.header("Docu-Bot 🤖")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

# initialize session state to persist the chat history. 
# Once the user exits the app however, the chat history will be lost.
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        # returns {query, result, source_documents}
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        # chat_history is a list of tuples: [("human", "prompt"), ("ai", "response")]

        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)