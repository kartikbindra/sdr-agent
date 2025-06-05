import streamlit as st
from agent_runner import run_agent

st.set_page_config(page_title="AI SDR Agent", page_icon="🤖")

st.title("🤖 AI SDR Agent for Inbound Leads")
st.markdown("Ask a question or type a message as if you're an inbound lead!")

user_input = st.text_input("Your Message:", key="user_input")

if user_input:
    with st.spinner("Thinking..."):
        response = run_agent(user_input)
        # response= "Hello"

    st.markdown("### Agent Response:")
    st.write(response)

    # if response["tool_messages"]:
    #     st.markdown("### Additional Info / Media")
    #     for msg in response["tool_messages"]:
    #         # Try to auto-detect and render links or videos
    #         if "youtube.com" in msg or "youtu.be" in msg:
    #             st.video(msg.strip())
    #         else:
    #             st.markdown(msg)