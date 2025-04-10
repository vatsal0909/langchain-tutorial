from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
import streamlit as st
from langchain_core.documents import Document
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
model  = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
import os
from pinecone import Pinecone
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
from langchain_pinecone import PineconeVectorStore
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("chatpdf")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


st.title("Chat with PDF")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "jpg", "png", "pdf"])

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=300, 
chunk_overlap=0,
)


prompt_template = PromptTemplate(
    template=' you are a company assistant chatbot, you havce to answser the given question->{question} based on the context provided, context={context}. if there is not answeer present in the context then simply reply ->No data found.',
    input_variables=['question','context']
)
parserr = StrOutputParser()


if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = list(loader.lazy_load())
    vector_store.add_documents(docs)
    
  
        
        
         



 
    
    question = st.text_input("write your query")
    
   

    if st.button("get result"):
         context = vector_store.similarity_search(question,k=5)
        
    
         chain = prompt_template|model|parserr
         
         response = chain.invoke({'question':question,'context':context})
         st.write("found on page number :",context[0].metadata['page'])
         st.success(response)
     
             
             
    


   




    
        

    

