#__import__('pysqlite3')
import sys

#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
from pathlib import Path
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be thorough in your answer and cite an relevant sections or rule numbers to support your answer.
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

st.set_page_config(page_title="PostGame Extract", page_icon=":tada:", layout="wide")
st.title(":blue[Post]:red[Game] Extract")
st.subheader('Interactive Sports Document OCR Tool', divider='red')
with st.expander('Instructions'):
    st.markdown(
        """
        #### Getting Started
        1. To get started with the app, please use the file uploader to upload an image of a document or pdf     
        (the tool currently only supports single page uploads).
          * If the sidebar containing the file uploader is not showing, please click the arrow in the top left corner 
                         (on mobile you may have to scroll up)
        2. Upon upload of your file:
          * An interactive chat will appear below. You may use the chat to ask specific questions about the document.
          * An image of the file will be available for view
          * Several export options will be made available by selecting an option from the drop-down on the left. After export is generated, please click the download button.
        - Export Options: 
          - For Image Files (.jpg, .jpeg, .png) 
            - Raw Text Extraction
            - Converted Scanned Image* (converts raw image of document into a digital scan). Best used for handwritten notes, and printed templates and forms containing handwriting.
            - Table Extractor to Excel (PDFs containing tabular data can be automatically converted to excel for further analysis)
          - For PDF Files
            - Raw Text Extraction
            - Table Extractor to Excel 
        ###### *please note scan conversion is only available for image files uploaded, and not available for PDFs.
        """
    )
st.subheader('', divider='blue')

left_column, right_column = st.columns(2)
st.sidebar.markdown('### About')
st.sidebar.markdown(
    'This app has been designed to allow users the ability to capture a photo of a variety of documents relating to sports. These documents may include anything from boxscores, play-by-play sheets, and even handwritten notes. PDF files are also supported.')
file_upload = st.sidebar.file_uploader("Please Upload a File", type=['jpg', 'jpeg', 'png', 'pdf'])

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

if prompts := st.chat_input("Ask me a question about your document!"):
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


