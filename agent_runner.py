# creating a Sales Developmenr Representative (SDR) ai agent using LangChain and Groq
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


import os
from dotenv import load_dotenv
load_dotenv()

### print the environment variables
# print("Environment Variables:")
# for key, value in os.environ.items():
#     print(f"{key}: {value}")
    

os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
os.environ["LANGSMTIH_TOKEN"] = os.getenv("LANGSMTIH_TOKEN")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langsmith.ai"
os.environ["LANGCHAIN_PROJECT"] = "pr-new-hobby-13"
os.environ["GROQ_API_KEY"] = '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'



loader = WebBaseLoader("https://docs.smith.langchain.com/")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})



# documents
texts

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langchain_docs",
    description="Useful for answering questions about LangSmith and LangChain.",
)

retriever_tool ### this is the tool we will use in the agent to answer questions about LangSmith and LangChain

retriever_tool.run("What is LangSmith?") ### this is how we can use the tool to answer questions about LangSmith and LangChain

### lead management tool
import json
from datetime import datetime

LEADS_FILE = "leads.json"

@tool
def save_lead_info(info: str) -> str:
    """
    Save structured lead info in the format:
    "Company: ABC Corp, Domain: Healthcare, Problem: Want better patient tracking, Budget: $10,000"
    """
    try:
        parsed = {}
        for line in info.split(","):
            key, val = line.strip().split(":", 1)
            parsed[key.strip().lower()] = val.strip()
        parsed["timestamp"] = str(datetime.now())

        with open(LEADS_FILE, "a") as f:
            f.write(json.dumps(parsed) + "\n")

        return "Lead information saved successfully!"
    except Exception as e:
        return f"Error saving lead info: {str(e)}"

### media trigger tool for LangChain and LangSmith demos
@tool
def show_demo_media(context: str) -> str:
    """
    Given a product use-case or industry context, return a video or image demo link.
    Example input: "Example Demo", "Langsmith Dashboard", "LangSmith for AI Agents Video"
    """
    media_map = {
        "langsmith dashboard": "https://youtu.be/LEIghyTGgQk?feature=shared",
        "langsmith for ai agents video": "https://youtu.be/gOK65vR0hIY?feature=shared",
        "Example demo": "https://youtu.be/kYtnLaJeia8?feature=shared",
    }

    for keyword, url in media_map.items():
        if keyword in context.lower():
            return f"Here’s a quick demo for {keyword}: {url}"

    return "Sorry, I couldn’t find a matching demo right now."

tools = [retriever_tool, save_lead_info, show_demo_media] ### set of tools we will use in the agent

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
Greet the user. Then ask for the following, one at a time: 
1. Company name 
2. Domain 
3. Problem statement 
4. Budget

Once you have all four, summarize and call the save_lead_info tool.

If they ask a product-related question, call the retriever_tool.
If they ask for a demo, call the show_demo_media tool.
If they ask anything other than these, respond witha friendly message acknowledging their question and offering to help with product info or demos.
""")

### setting up a memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

### selecting the chat model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

### binding tools to the agent chat model
model_with_tools = model.bind_tools(tools)

### agent executor

memory = MemorySaver()
agent_executor = create_react_agent(
    model=model_with_tools,
    prompt=prompt_template,
    tools=tools,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "sdr_agent"}}

### try agent
from langchain_core.messages import HumanMessage
response= agent_executor.invoke(
   { "messages":[
        HumanMessage(
            content="What is LangSmith? Can you also show me a demo of your product?"
        )
    ]}, config
)
# print(response["messages"]) ### this is the response from the agent

for chunk in agent_executor.stream (
   { "messages":[
        HumanMessage(
            content="hello I need help with my lead management. Can you help me save a lead?"
        )
    ]}, config
):
    print(chunk)
    print("----")
# print(response["messages"])

from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentExecutor
from langchain.agents import initialize_agent, AgentType

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor2 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# print(agent_executor2.run("hello I need help with my lead management. Can you help me save a lead?"))

from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

system_template = """
You are Ava, a friendly SDR AI voice assistant.
Your job is to greet the user, ask for their company name, domain, problem, and budget.
Answer product questions using the retriever_tool.
If they request a demo, use the show_demo_media tool.
Once all information is gathered, use save_lead_info and end the conversation politely.
"""

# Use the llm variable already defined, and pass your system_template as part of the prompt if needed.
# The initialize_agent function does not take a system_message argument directly.
# If you want to customize the system prompt, use the prompt argument in initialize_agent or PromptTemplate.

agent_executor2 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    prompt=prompt_template,  # Use the prompt_template defined earlier
        # You can add a custom prompt here if needed, e.g.:
    # prompt=PromptTemplate(input_variables=[], template=system_template)
)

# print(agent_executor2.run("hello I need help with my lead management. Can you help me save a lead?"))
# print(agent_executor2.run("What is LangSmith? Can you also show me a demo of your product?"))

# print(agent_executor2.run("I own a healthcare company called ABC Corp. We want to improve patient tracking. Our budget is $10,000."))

# print(agent_executor2.run("What is my lead information you stored?"))

def run_agent(user_input: str) -> str:
    """
    Run the agent with the given user input.
    """
    response = agent_executor2.run(user_input)
    return response

def run_agent_streaming(user_input: str) -> dict:
    from langchain_core.messages import HumanMessage

    final_text = ""
    tool_outputs = []

    # Stream config (optional)
    config = {"tags": ["streamlit_agent"]}

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    ):
        if "agent" in chunk:
            messages = chunk["agent"].get("messages", [])
            for msg in messages:
                if msg.content:
                    final_text += msg.content  # accumulate final agent messages

        if "tools" in chunk:
            messages = chunk["tools"].get("messages", [])
            for msg in messages:
                if msg.content:
                    tool_outputs.append(msg.content)

    return {
        "text": final_text.strip(),
        "tool_messages": tool_outputs
    }
