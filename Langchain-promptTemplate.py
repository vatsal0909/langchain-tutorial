from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate ,load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
load_dotenv()



st.title("Welcome to summary generator")

user_topic  = st.selectbox(
    'which topic would you like to get summary of',
 ('AWS bedrock', 'google gemini', 'GPT 4.0'))
user_paragraph_size  = st.selectbox(
    'how many pragraph you want',
 ('1-2 paragraph', '2-3 pargrph', '5 paragraphs'))
user_temprature  = st.selectbox(
    'which tone you want',
 ('scientific', 'some small code to invoke model', 'beginner friendly'))


prompt_template = load_prompt("template.json")
prompt = prompt_template.invoke({
    "topic":user_topic,
    "paragraph_size":user_paragraph_size,
    "paragraph_temprature":user_temprature
})



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

if st.button("summerize"):
    response = llm.invoke(prompt)
    st.write(response.content)
