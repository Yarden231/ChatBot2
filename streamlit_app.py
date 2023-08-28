import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

# Set the page configuration to match Retama's branding
st.set_page_config(page_title="Retama Support Chat", page_icon="🔒", layout="centered", initial_sidebar_state="auto", menu_items=None)

# Set OpenAI API key
openai.api_key = st.secrets.openai_key

# Title and Information
st.title("Retama Support Chat 🛡️")
st.info("Welcome to Retama's official support chat. We specialize in CRM and ERP solutions. How can we assist you today?", icon="🔒")

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you with Retama's services today?"}
    ]

# Load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Initializing Retama's knowledge base..."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True) 
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on Retama's CRM and ERP solutions. Your job is to answer queries related to these services."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat interface
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate assistant's response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
