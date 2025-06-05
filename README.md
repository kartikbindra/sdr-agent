This is a SDR Agent that leverages LangChain, GROQ and Llama LLM models.
it is a agent that has the knowledge of langsmith docs and can provide demo videos and information about langsmith and langchain too.

This is not a very hyper-tuned llm model and may not work right sometimes.

the main files are:
server.py (streamlit app)
agent -> app.ipynb
agent_runner.py -> agent wrapper file

use requirements.txt to install all the reqrd libraries
and, run `streamlit run server.py` to open the local streamlit server for testing the app