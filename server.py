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
    st.markdown(f"**AI Response:** {response['text']}")

    # Media rendering
    for media in response.get("media", []):
        if media["type"] == "image":
            st.image(media["url"], caption="Image Provided")
        elif media["type"] == "video":
            st.video(media["url"])
        elif media["type"] == "pdf":
            st.markdown(f"[View PDF]({media['url']})")
