from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
import streamlit as st
import tempfile
model  = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
import os
from pinecone import Pinecone
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
from langchain_pinecone import PineconeVectorStore
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("chatpdf")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


st.title("Chat with PDF")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "jpg", "png", "pdf"])

if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.lazy_load()

    
        

    

