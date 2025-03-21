__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb.utils.embedding_functions as embedding_functions
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
import os
import io
import numpy as np
#import imutils
import chromadb
import chromadb.api
import docx2txt
from dotenv import load_dotenv



chroma_client = chromadb.Client()

load_dotenv()
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
chromadb.api.client.SharedSystemClient.clear_system_cache()
# from streamlit_chromadb_connection import ChromadbConnection


# Calling required API keys from streamlit secrets
#OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
# Initializing extraction tools

try:
    rules_collection = chroma_client.get_collection('NFHS_Basketball_Rules', embedding_function=google_ef)
except:
    text = docx2txt.process('./ncboa rule book copy.docx').strip()
    rulebook_text = text.replace("\n", " ")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    splits = text_splitter.split_text(rulebook_text)
    rules_collection = chroma_client.create_collection('NFHS_Basketball_Rules', embedding_function=google_ef)
    ids = [f'id{i}' for i in range(0, len(splits))]
    rules_collection.add(documents=splits, ids=ids)

# Creating functions needed for workflow execution:
# Text Extraction

prompt_text = """
    You are an expert on high school level (NFHS) basketball rules available to assist referees answer questions
    pertaining to the rules. Use the following pieces of retrieved context to answer the question. Only use the informaiton
    provided in the context to produce an answer, and do not use outside knowledge. If you don't know 
    the answer, just say that you don't know. Be thorough in your answer and cite an relevant sections 
    or rule numbers to support your answer.
    Human: {question} 
    Context: {context} 
    Assistant:
    """

prompt = ChatPromptTemplate.from_template(template=prompt_text
                                          )
llm = ChatOpenAI(model='gpt-4o-mini')
# functions for chatbot graph
memory = MemorySaver()


class State(TypedDict):
    question: str
    context: list
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = rules_collection.query(query_texts=[state["question"]], n_results=3)
    return {"context": retrieved_docs['documents']}


def generate(state: State):
    docs_content = ''.join(state["context"][0])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
# Response generation function


def generate_response(query: str):
    response = graph.invoke({"question": query}, config=config)
    return response['answer']


# Scan tool functions
# Point ordering function used in perspective transform fucntion



# Page configurations and setup

st.set_page_config(page_title="RefPrep", page_icon=":tada:", layout="wide")
st.title(":blue[Ref]:red[Prep]")
st.subheader('Your Personal Rule Expert', divider='red')
with st.expander('Instructions'):
    st.markdown(
        """
        #### Getting Started
        To get started, simply ask RefPrep a question below. The assistant will review the NFHS rules book and provide
         a response along with relevant sections for reference. 
         * Leave your rule book in your bag
         * Receieve reliable responses that can be cross referenced with specific sections of the rule book    
         * Get answers to your questions pre-game, mid-game, or at any time you have a question about rules
        
        """
    )
st.subheader('', divider='blue')

left_column, right_column = st.columns(2)

# Initializing messages and memory in the session state for call back to chat history during chat session
if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# for message in st.session_state.messages:
# with st.chat_message(message["role"]):
# st.markdown(message["content"])

# If statement used to check for file upload to limit further code execution until after file has been uploaded.

    # try:
    # vectorstore.delete(vectorstore.get()['ids'])
    # except:
    # pass

st.markdown('#### Chatbot')
# Temporary file saving


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_history = []

# Chat configuration and setup

if prompts := st.chat_input("Ask me a question about NFHS rules!"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompts)
        st.session_state.messages.append({"role": "user", "content": prompts})
    with st.spinner("Generating an answer..."):
        with st.chat_message("assistant"):
            # message_placeholder = st.empty()
            # full_response = ""
            response = generate_response(prompts)
            st.markdown(response)
            # for response in response:
            # full_response += response
            # message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    # Deleting all items from vectorstore to avoid document hallucination/confusion
    # vectorstore.delete(vectorstore.get()['ids'])


