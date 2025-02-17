import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load groq API key
GROQ_API_KEY = "API_KEY"
groq_api_key = GROQ_API_KEY

if "final_documents" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://www.geeksforgeeks.org/dsa-tutorial-learn-data-structures-and-algorithms/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq APP")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

prompt_template = """
Answer the questions based on the context provided only.
Please provide the most accurate response based on the question
<context>
{context}
Questions:{input}
"""

document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(prompt_template))
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Input what you want to know")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print("Response Time:", time.process_time() - start)
    st.write(response['answer'])

    # Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------")
